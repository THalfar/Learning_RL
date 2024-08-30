import gymnasium as gym
import optuna
from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import EvalCallback, CallbackList 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, sync_envs_normalization
import os
import signal
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time




def create_env(env_id, n_envs=1):
    def make_env():
        return gym.make(env_id)
    return SubprocVecEnv([make_env] * n_envs)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

    env = create_env('HandManipulateBlockRotateParallelDense-v1', 23)

    # Define the evaluation callback
    eval_callback = EvalCallback(eval_env=env,
                                best_model_save_path='./models/transfer_test',
                                log_path='./logs/',
                                eval_freq=6e5 // 23,
                                deterministic=True,
                                render=False,
                                verbose=1,
                                n_eval_episodes=500)

    # Create a callback list with the evaluation callback
    callback_list = CallbackList([eval_callback])

    # Train the model with the callback list


    model = SAC.load('./models/callback_8_21_simple_0_best/best_model', env)    
    

    model.learn(total_timesteps=5e6, callback=callback_list, progress_bar=True)