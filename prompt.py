# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import utils
from collections import OrderedDict
from modules import Actor, Critic, RandomShiftsAug

        
class ResNet(nn.Module):
    def __init__(self, cfg, obs_shape):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.crop_size = obs_shape[-1]
        self.image_channel = 3
        self.repr_dim = 1024
        
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()

        model = self.setup_prompt(model)
        self.setup_grad(model)
        self.setup_head(cfg)
        
    def setup_grad(self, model):
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
        
        for k, p in self.frozen_layers.named_parameters():
            p.requires_grad = False

    def setup_prompt(self, model):
        self.prompt_location = self.cfg.location        
        self.num_tokens = self.cfg.num_tokens

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
        
        return model

    def setup_head(self, cfg):
        sample = torch.randn([1, 9, self.crop_size, self.crop_size])
        out_shape = self.forward_conv(sample).shape
        self.out_dim = out_shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.out_dim, self.repr_dim),
            nn.LayerNorm(self.repr_dim),
            nn.ReLU(inplace=True)
        )

    def incorporate_prompt(self, x):
        B = x.shape[0]
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
        x = self.prompt_layers(x)
        return x

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        if self.image_channel == 4:
            obs = torch.cat((obs[:, :3, :, :] / 255.0, obs[:, 3:, :, :]), dim=1)
        else:
            obs = obs / 255.0
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        obs = self.incorporate_prompt(obs)
        obs = self.frozen_layers(obs)

        conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        conv_current = conv[:, 1:, :, :, :]
        conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv
    
    def forward(self, x, return_feature=False):
        if self.frozen_layers.training:
            self.frozen_layers.eval()

        x = self.forward_conv(x)

        if return_feature:
            return x

        return self.head(x)

class PromptAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, 
                 prompt):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        print(f'prompt cfg: {prompt}')
        self.prompt_cfg = prompt

        # models
        self.encoder = ResNet(prompt, obs_shape).to(device)
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
        obs = obs.unsqueeze(0)

        obs = self.encoder(obs)
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
        
        obs = torch.cat((obs, aug_obs), dim=0)
        action = torch.cat((action, action), dim=0)
        target_Q = torch.cat((target_Q, target_Q), dim=0)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

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
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        aug_obs = utils.random_conv(original_obs)

        obs = self.encoder(obs)
        aug_obs = self.encoder(aug_obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs))
        metrics.update(self.update_actor(obs.detach(), step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
