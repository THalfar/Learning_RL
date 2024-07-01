import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import optuna
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import moviepy.editor as mpy
import tensorflow as tf
import time 
import os
import gc
import torch

# Varmistetaan, että GPU on käytössä TensorFlow:ssa
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available")
    for device in physical_devices:
        print(device)
else:
    print("No GPU available, using CPU")

start_time = time.time()
env = make_vec_env(lambda: gym.make("HumanoidStandup-v4"), n_envs=6)
model = PPO('MlpPolicy',env,verbose=0)
model.learn(total_timesteps=200000)

# Arviointi ja renderöinti
eval_env = gym.make("HumanoidStandup-v4", render_mode='rgb_array')
obs, info = eval_env.reset()
total_reward = 0.0
frames = []
for _ in range(420):
    frame = eval_env.render()
    if frame is not None:
        frames.append(frame)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = eval_env.step(action)
    total_reward += reward
    if done or trunc:
        break

eval_env.close()
end_time = time.time() - start_time
print(f'Baseline finished in {end_time/60:.2f} min.')
print(f'Total reward for baseline: {total_reward:.0f}')
clip = mpy.ImageSequenceClip(frames, fps=30)
clip.write_videofile(f'/workspace/videos/humanoid_rising_trial_baseline_{total_reward:.0f}.mp4', codec='libx264')
del model
del env
del eval_env
del obs
del frames
gc.collect()
torch.cuda.empty_cache()

best_reward = 0

def objective(trial):

    global best_reward

    start_time = time.time()

    n_steps = trial.suggest_int('n_steps', 32, 10000, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 1.0)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    total_timesteps = trial.suggest_int('total_timesteps', 6000, 1e6, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    
    
    network_type = trial.suggest_categorical('network_type', ['simple', 'advanced', 'medium', 'small', 'large'])
    
    if network_type == 'simple':
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    elif network_type == 'medium':
        policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    elif network_type == 'small':
        policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    elif network_type == 'large':
        policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    else:
        policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])

    # Käytetään DummyVecEnv:iä ja headless-tilaa
    env = make_vec_env(lambda: gym.make("HumanoidStandup-v4"), n_envs=6)
    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=lr,
        policy_kwargs=policy_kwargs,
        verbose=0,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        clip_range=clip_range

    )
    model.learn(total_timesteps=total_timesteps)

    # Arviointi ja renderöinti
    eval_env = gym.make("HumanoidStandup-v4", render_mode='rgb_array')
    obs, info = eval_env.reset()
    total_reward = 0.0
    frames = []
    for _ in range(420):
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        total_reward += reward
        if done or trunc:
            break

    eval_env.close()
    end_time = time.time() - start_time
    print(f'Trial {trial.number} finished in {end_time/60:.2f} min.')

    if total_reward > best_reward:        
        best_reward = total_reward
        print(f'New best reward: {best_reward:.0f}')
        clip = mpy.ImageSequenceClip(frames, fps=30)
        clip.write_videofile(f'/workspace/videos/humanoid_rising_best_{trial.number}_{total_reward:.0f}.mp4', codec='libx264')

    del model
    del env
    del eval_env
    del obs
    del frames
    gc.collect()
    torch.cuda.empty_cache()
        

    return total_reward

study = optuna.create_study(direction='maximize')
study.sampler = optuna.samplers.TPESampler(n_startup_trials=5)
study.optimize(objective, n_trials=4200)

best_params = study.best_params
print("Parhaat parametrit:", best_params)
