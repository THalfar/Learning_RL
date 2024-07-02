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
    model_path = f"./models/trial_{original_trial_number}.zip"
    # print(f"Saving model for trial {original_trial_number} to {model_path}")
    trial.set_user_attr(model_path, model_path)
    trial.set_user_attr('timesteps', timesteps)
    model.save(model_path)
    

def load_model(trial, env):
    
    # FIX for the key=value bug 
    for key, value in trial.user_attrs.items():
        if key == value and key.startswith('./models/'):
            model_path = value

    if os.path.exists(model_path):
        print(f"Loading existing model for trial {trial.number} from {model_path}")
        model = SAC.load(model_path, env)
        timesteps = trial.user_attrs.get('timesteps', 0)
        return model, timesteps
    return None, 0


def train_and_evaluate(params, total_timesteps, trial=None):

    env = create_env("Humanoid-v4", n_envs=16)
    params['policy_kwargs'] = build_policy_kwargs(params)
    
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
    
    rewards = []

    total_time = time.time()
    
    while timesteps < total_timesteps:

        learn_time_start = time.time()        
        model.learn(total_timesteps=20000, reset_num_timesteps=False)
        timesteps += 20000
        learn_time_stop = time.time() - learn_time_start
        
        eval_time_start = time.time()
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
        eval_time_stop = time.time() - eval_time_start        
        rewards.append(reward)
        mean_reward = np.mean(rewards[-3:])

        print(f'Trial {trial.number} timestep {timesteps} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {mean_reward:.1f} training time: {int(learn_time_stop)} sec. Eval time: {int(eval_time_stop)} sec.')        

        trial.report(mean_reward, timesteps)
        
        if trial.should_prune():
            print(f'Trial {trial.number} pruned at timestep {timesteps}.')                
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 240:
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
    
    save_model(model, trial, timesteps)
    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
    print(f'Total mean reward this training: {np.mean(rewards):.1f}')
    print(f'Total timesteps: {timesteps}')
    
    env.close()
    del model
    del env
    gc.collect()
    torch.cuda.empty_cache()
    
    return mean_reward

def save_video_of_model(params, trial, total_timesteps, name='base', steps = 40000):
    original_trial_number = trial.user_attrs.get('original_trial_number', trial.number)
    
    env = create_env("Humanoid-v4", n_envs=16)
    params['policy_kwargs'] = build_policy_kwargs(params)

    best_model_path = f"./models/{name}_trial_{original_trial_number}.zip"
    best_reward = -float('inf')
    
    model, timesteps = load_model(trial, env)
    if model is None:
        print(f'Model is None, creating new model for trial {trial.number}.')
        model = SAC(
            'MlpPolicy',
            env,
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            learning_rate=params['lr'] * 0.5,
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
    
    for timestep in range(timesteps, total_timesteps, steps):
        train_time = time.time()
        model.learn(total_timesteps=timestep, reset_num_timesteps=False)
        train_time_stop = time.time() - train_time

        eval_time = time.time()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=200, deterministic=True)
        eval_time_stop = time.time() - eval_time
        
        print(f'Timestep {int(timestep)} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} in {train_time_stop:.1f} sec. Eval time: {eval_time_stop:.1f} sec.')        
        if mean_reward > best_reward:
            print(f'New best reward: {mean_reward:.2f} > {best_reward:.2f} saving model.')
            best_reward = mean_reward
            model.save(best_model_path)

        
    env.close()
    
    eval_env = gym.make("Humanoid-v4", render_mode='rgb_array')  
    obs, info = eval_env.reset()
    frames = []    
    total_reward = 0.0
    model.load(best_model_path)

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
    clip.write_videofile(f'/workspace/videos/humanoid_walking_{name}_trial_{trial.number}_{total_reward:.0f}.mp4', codec='libx264')
    
    del model
    del env
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_reward

def objective(trial):

    total_timesteps = int(2e5)

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

previous_study = optuna.load_study(study_name='7_01_sac_slope', storage='sqlite:///gymnasium_humanoid_walking.db')

last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
                    key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:20]

storage = optuna.storages.RDBStorage('sqlite:///gymnasium_humanoid_walking.db')
study = optuna.create_study(
    study_name='7_02_sac_walking_best_slope_load_test1',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling = False)
)

for idx, trial in enumerate(last_trials):
    print(f'Trial {trial.number} value {trial.value} user attrs {trial.user_attrs}')
    study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)

study.optimize(objective, n_trials=100)

top_trials = sorted([t for t in study.trials if t.state != optuna.trial.TrialState.PRUNED], key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:3]

for trial in top_trials:
    print(f"Training top trial {trial.number} with long training duration...")
    print(f'Value: {trial.value:.5f}')
    print(f'Params: {trial.params}')
    
    params = trial.params
    params['policy_kwargs'] = build_policy_kwargs(params)

    total_reward = save_video_of_model(params, trial, total_timesteps=int(1e6), name=study.study_name)
    print(f"Video saved for trial {trial.number} with total reward: {total_reward:.2f}")
