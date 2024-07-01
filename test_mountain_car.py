import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import optuna
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import moviepy.editor as mpy
import tensorflow as tf

# env = gym.make('MountainCar-v0', render_mode='rgb_array')
# env.reset()
# frame = env.render()
# print(frame)
# env.close()


class CustomMountainCarEnv(gym.Env):
    def __init__(self):
        super(CustomMountainCarEnv, self).__init__()
        self.env = gym.make('MountainCar-v0', render_mode='rgb_array')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        position, velocity = obs
        if velocity > 0:
            reward += velocity * 10
        reward += position + +.5
        # reward += (velocity * 10)**2
        return obs, reward, done, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# Varmistetaan, että GPU on käytössä TensorFlow:ssa
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available")
    for device in physical_devices:
        print(device)
else:
    print("No GPU available, using CPU")

def objective(trial):
    n_steps = trial.suggest_int('n_steps', 2, 2000, log = True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log = True)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log = True)

    env = make_vec_env(lambda: CustomMountainCarEnv(), n_envs=10)
    model = PPO('MlpPolicy', env, n_steps=n_steps, gamma=gamma, learning_rate=lr, verbose=0)
    model.learn(total_timesteps=100000)

    eval_env = CustomMountainCarEnv()
    obs, info = eval_env.reset()
    total_reward = 0.0
    frames = []
    for _ in range(200):
        frame = eval_env.render()  
        # print(frame)      
        if frame is not None:
            frames.append(frame)        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truc , info = eval_env.step(action)
        total_reward += reward
        if done:
            break

    eval_env.close()
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(f'/workspace/videos/trial_{trial.number}.mp4', codec='libx264')

    return total_reward


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=42)

best_params = study.best_params
print("Parhaat parametrit:", best_params)
