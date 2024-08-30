import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import moviepy.editor as mpy
import tensorflow as tf
import time
import gc
import torch
import os
import signal
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

env_name = 'HandManipulateBlockRotateParallelDense-v1'

eval_env = gym.make(env_name, render_mode='rgb_array')  


model = SAC.load('./models/callback_8_23_long_0_best/best_model', env=eval_env)

num_rounds = 1000
num_success = 0
frames = []

for round in range(num_rounds):
    obs, info = eval_env.reset()
    total_reward = 0
    

    for _ in range(100):
        frame = eval_env.render()
        if frame is not None and round % 10 == 0:
            frames.append(frame)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        total_reward += reward
        if info.get('is_success', False):
            print('Success')
            num_success += 1
            break

        if done:
            print('Done')            
            break 

    # eval_env.close()
    print(f'Total reward: {total_reward}')

clip_name = f'/workspace/handvideos/collage_100.mp4'
clip = mpy.ImageSequenceClip(frames, fps=30)
clip.write_videofile(clip_name, codec='libx264')
print(f'Video saved to {clip_name}')

print(f'Success rate {100 * (num_success/num_rounds):.2f}%')
del model
del eval_env