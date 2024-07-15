import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
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

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Stopping the study...')
    study.stop()
    exit(0)



def create_env(env_id, n_envs=1):
    def make_env():
        return gym.make(env_id)
    return SubprocVecEnv([make_env] * n_envs)


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
    
    model_path = f"./models/{trial.study.study_name}_{trial.number}.zip"
    print(f"Saving model for trial to {model_path}")
    trial.set_user_attr('model_path', model_path)
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

    env = create_env("HumanoidStandup-v4", n_envs=22)
    
    model, timesteps = load_model(trial, env)
    if model is None:
        model = SAC('MlpPolicy',env,
            batch_size=params['batch_size'],
            gamma = 1 - params['gamma_eps'],
            learning_rate=params['lr'],            
            verbose=0,
            ent_coef= f'auto_{params["ent_start"]}',
            tau=params['tau'],
            buffer_size=params['buffer_size']
            )
    timesteps = 0
    
    rewards_zero = [0]
    rewards = []
    total_time = time.time()
    steps = 25000
    
    while True:

        learn_time_start = time.time()        
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        timesteps += steps
        learn_time_stop = time.time() - learn_time_start

        # Breaks if total timesteps reached so that last evaluation is done
        if timesteps >= total_timesteps:
            break
        
        eval_time_start = time.time()
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
        eval_time_stop = time.time() - eval_time_start        
        rewards_zero.append(reward)
        rewards.append(reward)
        running_mean = np.mean(rewards[-3:])
        total_mean = np.mean(rewards)
        slope, _ = np.polyfit(range(len(rewards_zero)), rewards_zero, 1)

        print(f'Timestep {timesteps} slope: {slope:.2f} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {running_mean:.1f} total mean: {total_mean:.1f} training time: {int(learn_time_stop)} sec with {steps/learn_time_stop:.1f} fps. . Eval time: {int(eval_time_stop)} sec.')        

        trial.report(slope, timesteps)
        
        if trial.should_prune():
            print(f'Trial {trial.number} pruned at timestep {timesteps}.')                
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 120:
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
    
    eval_time_start = time.time()
    reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
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
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    
    total_timesteps = int(1e5)

    params = {
        'batch_size': trial.suggest_int('batch_size', 64, 3000, log=True),
        'gamma_eps' : trial.suggest_float('gamma_eps', 1e-5, 1e-2, log=True),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),        
        'tau': trial.suggest_float('tau', 1e-6, 0.1, log=True),
        'ent_start' : trial.suggest_float('ent_start', 0.1, 0.9),
        'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(1e6), log=True),
        }

    
    reward = train_and_evaluate(params, total_timesteps=total_timesteps, trial=trial)

    return reward

if __name__ == '__main__':
        
    storage = optuna.storages.RDBStorage('sqlite:///gymnasium_standup.db')
    study = optuna.create_study(
        study_name='7_14_sac_slope',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_min_trials=20),
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.CmaEsSampler(warn_independent_sampling = False)
    )

    signal.signal(signal.SIGINT, signal_handler)
    study.optimize(objective, n_trials=142000)
