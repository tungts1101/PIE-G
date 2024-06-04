# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms, models
import utils
from collections import OrderedDict
import vision_transformer as vits


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Segmentor(nn.Module):
    def __init__(self):
        super().__init__()
        model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
        
        self.model = model
        self.patch_size = 8
    
    @torch.no_grad()
    def forward(self, x, head=2):
        B, _, H, W = x.shape
        w, h = W - W % self.patch_size, H - H % self.patch_size
        x = x[:, :, :h, :w]

        h_featmap = H // self.patch_size
        w_featmap = W // self.patch_size
        
        attentions = self.model.get_last_selfattention(x)
        # we keep only the [CLS] token
        attentions = attentions[:, head, 0, 1:].reshape(B, 1, h_featmap, w_featmap)
        attentions = nn.functional.interpolate(attentions, size=(H, W), mode="nearest")
        return attentions
        
class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, cfg, obs_shape, device):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.crop_size = obs_shape[-1]
        self.image_channel = 3
        self.repr_dim = 1024

        self.segmentor = Segmentor()
        self.segmentor.eval()
        
        model_type = cfg.model_type
        model = self.get_pretrained_model(model_type)

        model = self.setup_prompt(model)
        self.setup_grad(model)
        self.setup_head(cfg)
        
        self.segmentor.to(device)

    def setup_grad(self, model):
        if self.cfg.location == "below":
            self.prompt_layers = nn.Sequential(OrderedDict([
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
            ]))
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("layer1", model.layer1),
                ("layer2", model.layer2),
            ]))
            self.tuned_layers = nn.Identity()
        elif self.cfg.location == "pad":
            self.prompt_layers = nn.Identity()
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
                ("layer1", model.layer1),
                ("layer2", model.layer2)
            ]))
            self.tuned_layers = nn.Identity()
        elif self.cfg.location == "concat":
            self.prompt_layers = nn.Sequential(OrderedDict([
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
            ]))
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("layer1", model.layer1),
                ("layer2", model.layer2),
            ]))
            self.tuned_layers = nn.Identity()
        
        for k, p in self.frozen_layers.named_parameters():
            p.requires_grad = False

    def setup_prompt(self, model):
        # ONLY support below and pad
        self.prompt_location = self.cfg.location        
        self.num_tokens = self.cfg.num_tokens
        if self.prompt_location == "below":
            return self._setup_prompt_below(model)
        elif self.prompt_location == "pad":
            return self._setup_prompt_pad(model)
        elif self.prompt_location == "concat":
            return self._setup_prompt_concat(model)
        else:
            raise ValueError(
                "ResNet models cannot use prompt location {}".format(
                    self.prompt_location))

    def _setup_prompt_concat(self, model):
        # modify first conv layer
        old_weight = model.conv1.weight  # [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(
            4, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        torch.nn.init.xavier_uniform(model.conv1.weight)

        model.conv1.weight[:, :3, :, :].data.copy_(old_weight)
        return model
    
    def _setup_prompt_below(self, model):
        if self.cfg.initiation == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.crop_size, self.crop_size
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])/3] * self.num_tokens,
                std=[sum([0.229, 0.224, 0.225])/3] * self.num_tokens,
            )

        elif self.cfg.initiation == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.crop_size, self.crop_size
            ))

            nn.init.normal_(self.prompt_embeddings.data)

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        # modify first conv layer
        old_weight = model.conv1.weight  # [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(
            self.num_tokens+3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        torch.nn.init.xavier_uniform(model.conv1.weight)

        model.conv1.weight[:, :3, :, :].data.copy_(old_weight)
        return model

    def _setup_prompt_pad(self, model):
        if self.cfg.initiation == "random":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.crop_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.crop_size, 2 * self.num_tokens
            ))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        elif self.cfg.initiation == "gaussian":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.crop_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.crop_size, 2 * self.num_tokens
            ))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Other initiation scheme is not supported")
        return model

    def get_pretrained_model(self, model_type):
        model_root = self.cfg.model_root
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)   # 512
        model.fc = nn.Identity()
        return model

    def setup_head(self, cfg):
        sample = torch.randn([1, 9, self.crop_size, self.crop_size])
        out_shape = self.forward_conv(sample).shape
        self.out_dim = out_shape[1]
        
        self.repr_dim = self.out_dim
        self.head = nn.LayerNorm(self.repr_dim)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        if self.prompt_location == "below":
            x = torch.cat((
                    x,
                    self.prompt_norm(
                        self.prompt_embeddings).expand(B, -1, -1, -1),
                ), dim=1)
        elif self.prompt_location == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.num_tokens, :],
                x, prompt_emb_tb[:, :, self.num_tokens:, :]
            ), dim=-2)
            # (B, 3, crop_size + num_prompts, crop_size + num_prompts)
        elif self.prompt_location == "concat":
            x = torch.cat((x, self.segmentor(x)), dim=1)
        else:
            raise ValueError("not supported yet")
        x = self.prompt_layers(x)
        return x

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        obs = self.incorporate_prompt(obs)
        obs = self.frozen_layers(obs)

        conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        # conv_current = conv[:, 1:, :, :, :]
        # conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        # conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv
    
    def forward(self, x, return_feature=False):
        x = self.get_features(x)

        if return_feature:
            return x

        return self.head(x)

    def get_features(self, x):
        """get a (batch_size, 2048) feature"""
        if self.frozen_layers.training:
            self.frozen_layers.eval()
        
        x = self.forward_conv(x)

        return x

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class PromptAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, 
                 prompt, svea_alpha, svea_beta):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.svea_alpha = svea_alpha
        self.svea_beta = svea_beta
        
        print(f'prompt cfg: {prompt}')

        # models
        self.encoder = ResNet(prompt, obs_shape, device).to(device)
        utils.log_model_info(self.encoder)

        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
    
    def save(self):
        return {
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),}

    def load(self, state_dict_map):
        self.encoder.load_state_dict(state_dict_map['encoder'])
        self.actor.load_state_dict(state_dict_map['actor'])
        self.critic.load_state_dict(state_dict_map['critic'])
        self.critic_target.load_state_dict(state_dict_map['critic_target'])
        return self

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        if self.svea_alpha == self.svea_beta:
            obs = torch.cat((obs, aug_obs), dim=0)
            action = torch.cat((action, action), dim=0)
            target_Q = torch.cat((target_Q, target_Q), dim=0)

            Q1, Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))
        else:
            Q1, Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))
            Q1_aug, Q2_aug = self.critic(aug_obs, action)
            critic_loss += self.svea_beta * (F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)

        # strong augmentation
        aug_obs = self.encoder(utils.random_overlay(original_obs))

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
