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
from datetime import datetime

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


def train_and_evaluate(params, total_timesteps):

    env = create_env("Humanoid-v4", n_envs=16)
    params['policy_kwargs'] = build_policy_kwargs(params)

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
    
    for timestep in range(0, total_timesteps, int(1e4)):

        eval_time = time.time()
        model.learn(total_timesteps=timestep, reset_num_timesteps=False)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3, deterministic=True)
        eval_time_stop = time.time() - eval_time
        print(f'Timestep {int(timestep)} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} in {eval_time_stop:.2f} sec.')        

        if eval_time_stop > 500:  # Custom condition for pruning
            print(f'Exceeded limit with {eval_time_stop:.2f} sec. Pruning.')
            break
            
    eval_time = time.time()
    last_reward, last_std = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    eval_time_stop = time.time() - eval_time
    
    sum_neurons = sum([params[f'actor_layer_{i+1}_neurons'] for i in range(params['actor_network_depth'])])
    sum_neurons += sum([params[f'critic_layer_{i+1}_neurons'] for i in range(params['critic_network_depth'])])
    print(f'Last Mean reward: {last_reward:.2f} +/- {last_std:.2f} in {eval_time_stop:.2f} sec with {sum_neurons} neurons.')
    
    env.close()
    del model
    del env
    gc.collect()
    torch.cuda.empty_cache()

    
    return last_reward, last_std


def save_video_of_model(params, trial_num, total_timesteps):

    env = create_env("Humanoid-v4", n_envs=16)
    params['policy_kwargs'] = build_policy_kwargs(params)

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

    eval_time = time.time()
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    eval_time_stop = time.time() - eval_time
    print(f'Trial {trial_num} Mean reward: {mean_reward} +/- {std_reward} with {eval_time_stop/60:.2f} min.')
    env.close()
    
    eval_env = gym.make("Humanoid-v4", render_mode='rgb_array')  
    obs, info = eval_env.reset()
    frames = []    
    total_reward = 0.0

    for _ in range(1000):
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        total_reward += reward
        if done or trunc:
            break

    eval_env.close()

    
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(f'/workspace/videos/humanoid_walking_ga_trial_{trial_num}_{total_reward:.0f}.mp4', codec='libx264')
    
    del model
    del env
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_reward


def objective(trial):

    params = {
        'batch_size': trial.suggest_int('batch_size', 1000, 2000),
        'gamma': trial.suggest_float('gamma', 0.98, 0.999999),
        'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
        'total_timesteps': int(1e5),
        'tau': trial.suggest_float('tau', 1e-3, 1.0, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.3, 0.7),
        'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(1e6), log=True),
        'learning_starts': trial.suggest_int('learning_starts', 1000, 10000, log=True),
        'train_freq': trial.suggest_int('train_freq', 10, 60),
        'gradient_steps': trial.suggest_int('gradient_steps', 70, 150),        
        'actor_network_depth': trial.suggest_int('actor_network_depth', 1, 3),
        'critic_network_depth': trial.suggest_int('critic_network_depth', 1, 3)
    }

    for i in range(params['actor_network_depth']):
        params[f'actor_layer_{i+1}_neurons'] = trial.suggest_int(f'actor_layer_{i+1}_neurons', 32, 1024, log=True)

    for i in range(params['critic_network_depth']):
        params[f'critic_layer_{i+1}_neurons'] = trial.suggest_int(f'critic_layer_{i+1}_neurons', 32, 1024, log=True)
    
    mean_reward, std = train_and_evaluate(params, total_timesteps=params['total_timesteps'])

    return mean_reward, std


previous_study = optuna.load_study(study_name='6_25_sac_slope', storage='sqlite:///gymnasium_walking_humanoid.db')

last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
                    key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:10]


storage = optuna.storages.RDBStorage('sqlite:///gymnasium_walking_humanoid.db')
study = optuna.create_study(
    study_name='6_28_sac_ga_5',
    directions=['maximize','minimize'],
    sampler = optuna.samplers.NSGAIIISampler(),    
    storage=storage,
    load_if_exists=True
)

for idx, trial in enumerate(last_trials):
    params = trial.params
    distributions = trial.distributions  # Extract distributions from the previous trial directly

    # Re-evaluate or use the stored results to add trials to the new study
    mean_reward, std_reward = train_and_evaluate(params, total_timesteps=int(1e5))

    # Create a new FrozenTrial with the appropriate distributions and other data
    trial = optuna.trial.FrozenTrial(
    number=idx,
    values=[mean_reward, std_reward],  # Example final objective values
    datetime_start=datetime.now(),
    datetime_complete=datetime.now(),
    params={},  # Your actual parameters here
    distributions={},  # Your actual distributions here
    user_attrs={},
    system_attrs={},
    intermediate_values={},  # Any intermediate values if prese
    value= None,
    state=optuna.trial.TrialState.COMPLETE,
    trial_id=idx ,# Ensure unique trial IDs or fetch dynamically
    
    
    )
    print(f'Added trial {idx} with mean reward: {mean_reward:.2f} and std: {std_reward:.2f}')
    study.add_trial(trial)

    
    
study.optimize(objective, n_trials=420)

top_trials = sorted([t for t in study.trials if t.state != optuna.trial.TrialState.PRUNED],key=lambda t: t.value if t.value is not None else float('-inf'),reverse=True)[:3]

for trial in top_trials:
    print(f"Training top trial {trial.number} with long training duration...")
    print(f'Value: {trial.value:.5f}')
    print(f'Params: {trial.params}')
    
    params = trial.params
    params['policy_kwargs'] = build_policy_kwargs(params)

    total_reward = save_video_of_model(params, trial.number, total_timesteps=int(1e6), name = study.study_name)
    print(f"Video saved for trial {trial.number} with total reward: {total_reward:.2f}")

