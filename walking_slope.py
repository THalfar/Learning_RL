import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import optuna
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import moviepy.editor as mpy
import tensorflow as tf
import time
import gc
import torch
import os


print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available")
    for device in physical_devices:
        print(device)
else:
    print("No GPU available, using CPU")


def create_env(env_id, n_envs=1):
    def make_env():
        return gym.make(env_id)
    return DummyVecEnv([make_env] * n_envs)


def build_policy_kwargs(params):
    actor_network_depth = params['actor_network_depth']
    critic_network_depth = params['critic_network_depth']
    
    actor_arch = []
    critic_arch = []

    for i in range(actor_network_depth):
        num_neurons = params[f'actor_layer_{i+1}_neurons']
        actor_arch.append(num_neurons)

    for i in range(critic_network_depth):
        num_neurons = params[f'critic_layer_{i+1}_neurons']
        critic_arch.append(num_neurons)

    policy_kwargs = dict(net_arch=dict(pi=actor_arch, qf=critic_arch))
    return policy_kwargs


def save_model(model, trial, timesteps):
    original_trial_number = trial.user_attrs.get('original_trial_number', trial.number)
    model_path = f"./models/{trial.study.study_name}_{original_trial_number}.zip"
    print(f"Saving model for trial {original_trial_number} to {model_path}")
    trial.set_user_attr(model_path, model_path)
    trial.set_user_attr('timesteps', timesteps)
    model.save(model_path)
    

def load_model(trial, env):
    original_trial_number = trial.user_attrs.get('original_trial_number', trial.number)
    model_path = f"./models/{trial.study.study_name}_{original_trial_number}.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model for trial {trial.number} from {model_path}")
        model = SAC.load(model_path, env)
        timesteps = trial.user_attrs.get('timesteps', 0)
        return model, timesteps
    return None, 0

def train_and_evaluate(params, total_timesteps, trial=None):

    env = create_env("Humanoid-v4", n_envs=20)
    params['policy_kwargs'] = build_policy_kwargs(params)
   
    # Load existing model if it exists
    model, timesteps = load_model(trial, env)
    if model is None:
        model = SAC(
            'MlpPolicy',
            env,
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            learning_rate=params['lr'],
            policy_kwargs=params['policy_kwargs'],
            verbose=0,
            ent_coef=params['ent_coef'],
            tau=params['tau'],
            buffer_size=params['buffer_size'],
            learning_starts=params['learning_starts'],
            train_freq=params['train_freq'],
            gradient_steps=params['gradient_steps']
        )
        timesteps = 0
    
    rewards_zero = [0]
    rewards = []
    total_time = time.time()
    
    while True:

        learn_time_start = time.time()        
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        timesteps += 10000
        learn_time_stop = time.time() - learn_time_start

        # Breaks if total timesteps reached so that last evaluation is done
        if timesteps >= total_timesteps:
            break
        
        eval_time_start = time.time()
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30, deterministic=True)
        eval_time_stop = time.time() - eval_time_start        
        rewards_zero.append(reward)
        rewards.append(reward)
        running_mean = np.mean(rewards[-3:])
        total_mean = np.mean(rewards)
        slope, _ = np.polyfit(range(len(rewards_zero)), rewards_zero, 1)

        print(f'Timestep {timesteps} slope: {slope:.2f} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {running_mean:.1f} total mean: {total_mean:.1f} training time: {int(learn_time_stop)} sec. Eval time: {int(eval_time_stop)} sec.')        

        trial.report(slope, timesteps)
        
        if trial.should_prune():
            print(f'Trial {trial.number} pruned at timestep {timesteps}.')                
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 180:
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
    
    eval_time_start = time.time()
    reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    eval_time_stop = time.time() - eval_time_start        

    rewards_zero.append(reward)
    rewards.append(reward)
    running_mean = np.mean(rewards[-3:])
    total_mean = np.mean(rewards)
    slope, _ = np.polyfit(range(len(rewards_zero)), rewards_zero, 1)
    print(f'Last eval. Slope: {slope:.2f} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {running_mean:.1f} mean: {total_mean:.2f}. Train time: {learn_time_stop} sec. Eval time: {int(eval_time_stop)} sec.') 
        
    trial.set_user_attr('mean3', running_mean) 
    trial.set_user_attr('reward', reward)
    trial.set_user_attr('std', std_reward)
    trial.set_user_attr('total_mean', total_mean)

    save_model(model, trial, timesteps)    
    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
    print(f'Total mean reward this training: {np.mean(rewards):.1f}')
    print(f'Total timesteps: {timesteps}')
    
    env.close()
    del model
    del env
    gc.collect()
    torch.cuda.empty_cache()
    
    return slope


def objective(trial):

    total_timesteps = int(5e4)

    params = {
        'batch_size': trial.suggest_int('batch_size', 256, 2000, log=True),
        'gamma': trial.suggest_float('gamma', 0.98, 0.999999),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),        
        'tau': trial.suggest_float('tau', 1e-3, 1.0, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.1, 0.7),
        'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(1e6), log=True),
        'learning_starts': trial.suggest_int('learning_starts', 100, 10000, log=True),
        'train_freq': trial.suggest_int('train_freq', 10, 60),
        'gradient_steps': trial.suggest_int('gradient_steps', 70, 150),
        'actor_network_depth': trial.suggest_int('actor_network_depth', 1, 3),
        'critic_network_depth': trial.suggest_int('critic_network_depth', 1, 3)        
    }

    for i in range(params['actor_network_depth']):
        params[f'actor_layer_{i+1}_neurons'] = trial.suggest_int(f'actor_layer_{i+1}_neurons', 32, 1024, log=True)

    for i in range(params['critic_network_depth']):
        params[f'critic_layer_{i+1}_neurons'] = trial.suggest_int(f'critic_layer_{i+1}_neurons', 32, 1024, log=True)
    
    reward = train_and_evaluate(params, total_timesteps=total_timesteps, trial=trial)

    return reward


storage = optuna.storages.RDBStorage('sqlite:///gymnasium_humanoid_walking.db')
study = optuna.create_study(
    study_name='7_01_sac_slope',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.CmaEsSampler(warn_independent_sampling = False)
)

study.optimize(objective, n_trials=420)
