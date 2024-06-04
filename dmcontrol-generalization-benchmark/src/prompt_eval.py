import torch
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from video import VideoRecorder
import augmentations
from logger import Logger
import hydra

def obs_to_input(obs):
	if isinstance(obs, utils.LazyFrames):
		_obs = np.array(obs)
	else:
		_obs = obs
	_obs = torch.FloatTensor(_obs).cuda()
	_obs = _obs.unsqueeze(0)
	return _obs


def evaluate(env, agent, video, num_episodes, eval_mode, L):
	episode_rewards = []
	for i in tqdm(range(num_episodes)):
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			with torch.no_grad():
				action = agent.act(np.array(obs), int(1e6), eval_mode=True)

			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			obs = next_obs

		if L is not None:
			L.log(f'eval/episode_reward', episode_reward, i)
			L.dump(i)
		video.save(f'eval_{eval_mode}_{i}.mp4')
  
		episode_rewards.append(episode_reward)

	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	print('args.seed:', args.seed)
	args.image_size = 84
	args.image_crop_size = 84
	
	print(f'args: {args}')
 
	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		intensity=args.distracting_cs_intensity
	)

	# Set working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.prompt, str(args.seed))
	print('Working directory:', work_dir)
	assert os.path.exists(work_dir), f'specified working directory {work_dir} does not exist'
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
 
	# Check if evaluation has already been run
	if args.eval_mode == 'distracting_cs':
		results_fp = os.path.join(work_dir, args.eval_mode+'_'+str(args.distracting_cs_intensity).replace('.', '_')+'.pt')
	else:
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
	assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

	assert torch.cuda.is_available(), 'must have cuda enabled'

	L = Logger(work_dir)
 
	with torch.no_grad():
		snapshot = torch.load(f'{work_dir}/snapshot1.pt')
		agent = hydra.utils.instantiate(snapshot['cfg'])['agent']
		agent.load(snapshot['state_dict']).train(False)

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, L)
		print('Reward:', int(reward))
		L.log(f'eval/reward', reward)
		L.dump(args.eval_episodes)

if __name__ == '__main__':
	args = parse_args()
	main(args)
