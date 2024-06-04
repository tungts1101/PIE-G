import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
import math


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	"""Reserves memory for replay buffer"""
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
		self.capacity = capacity
		self.batch_size = batch_size

		self._obses = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
		else:
			self._obses[self.idx] = (obses)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses = [], []
		for i in idxs:
			obs, next_obs = self._obses[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
		return np.array(obses), np.array(next_obses)

	def sample_soda(self, n=None):
		idxs = self._get_idxs(n)
		obs, _ = self._encode_obses(idxs)
		return torch.as_tensor(obs).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_drq(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_shift(obs, pad)
		next_obs = augmentations.random_shift(next_obs, pad)

		return obs, actions, rewards, next_obs, not_dones

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns total number of params in a network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'


import time
class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

from torch.distributions.utils import _standard_normal
from torch import distributions as pyd
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
import re
def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

import torch.nn as nn
def mlp(input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        use_ln=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim)]
        if use_ln:
            mods += [nn.LayerNorm(hidden_dim), nn.Tanh()]
        else:
            mods += [nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


def random_conv(x):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)

def log_model_info(model, verbose=False):
    """Logs model info"""
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    print("tuned percent:%.3f"%(model_grad_params * 100/model_total_params))

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

# DINO related function
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor