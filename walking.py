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
    print(f"Saving model to {model_path}")
    trial.set_user_attr('model_path', model_path)
    trial.set_user_attr('timesteps', timesteps)
    model.save(model_path)
    

def load_model(trial, env):

    model_path = None
    
    # FIX for the key=value bug 
    if 'model_path' not in trial.user_attrs:
        for key, value in trial.user_attrs.items():
            if key == value and key.startswith('./models/'):
                model_path = trial.user_attrs.get('model_path', value)
                break         

    if model_path is None:
        model_path = trial.user_attrs.get('model_path', None)
        
    if  model_path is not None:
        if os.path.exists(model_path):
            print(f"Loading existing model for trial {trial.number} from {model_path}")
            model = SAC.load(model_path, env)
            timesteps = trial.user_attrs.get('timesteps', 0)
            return model, timesteps
        else:
            print(f'Model path {model_path} does not exist. Creating new model for trial {trial.number}.')
    
    return None, 0


def train_and_evaluate(params, total_timesteps, trial=None):

    gc.collect()
    torch.cuda.empty_cache()
    

    env = create_env("Hopper-v4", n_envs=20)
    # params['policy_kwargs'] = build_policy_kwargs(params)

    gamma = 1 - params['gamma_eps']
    
    model, timesteps = load_model(trial, env)
    if model is None:
        model = SAC(
            'MlpPolicy',
            env,
            batch_size=params['batch_size'],
            gamma=gamma,
            learning_rate=params['lr'],
            # policy_kwargs=params['policy_kwargs'],
            verbose=0,
            ent_coef=params['ent_coef'],
            tau=params['tau']
            # buffer_size=params['buffer_size'],
            # learning_starts=params['learning_starts'],
            # train_freq=params['train_freq'],
            # gradient_steps=params['gradient_steps']
        )
        timesteps = 0
    
    step = 20000
    
    rewards = []
    current_lr = params['lr']
    best_reward = -float('inf')

    total_time = time.time()
    
    while timesteps < total_timesteps:

        current_lr *= params['lr_reduction']
        model.lr_schedule = lambda _: current_lr
        print(f'New learning rate: {current_lr:.8f}')

        learn_time_start = time.time()        
        model.learn(total_timesteps=step, reset_num_timesteps=False)
        timesteps += step
        learn_time_stop = time.time() - learn_time_start
        
        eval_time_start = time.time()
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
        eval_time_stop = time.time() - eval_time_start        
        rewards.append(reward)
        mean_reward = np.mean(rewards[-3:])

        print(f'Trial {trial.number} timestep {timesteps} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {mean_reward:.1f} training time: {learn_time_stop:.1f} sec with {step/learn_time_stop:.1f} fps. Eval time: {eval_time_stop:.1f} sec.')        

        trial.report(mean_reward, timesteps)

        if reward > best_reward:
            print(f'New best reward: {reward:.2f} > {best_reward:.2f}')
            best_reward = reward
            trial.user_attrs['best_reward'] = best_reward
            save_model(model, trial, timesteps)
        
        if trial.should_prune():
            print(f'Trial {trial.number} pruned at timestep {timesteps}.')                
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 60:
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
    
    
    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
    print(f'Last mean reward: {mean_reward:.1f}')
    print(f'Total timesteps: {timesteps}')
    
    env.close()
    del model
    del env
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_reward

def sava_video_of_trial(trial, name = 'walker2d'):

    
    eval_env = gym.make("Hopper-v4", render_mode='rgb_array')  
    obs, info = eval_env.reset()
    frames = []    
    total_reward = 0.0    
    model, _ = load_model(trial, eval_env)
    
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
    clip.write_videofile(f'/workspace/videos/{name}_trial_{trial.number}_{total_reward:.0f}.mp4', codec='libx264')
    
    del model    
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_reward


def save_video_of_model(params, trial, total_timesteps,  steps = 40000, name='humanoid'):

    gc.collect()
    torch.cuda.empty_cache()
    
    env = create_env("Walker2d-v4", n_envs=20)
    # params['policy_kwargs'] = build_policy_kwargs(params)

    best_model_path = f"./models/{name}_trial_{trial.number}_long.zip"
    best_reward = -float('inf')
    
    model, timesteps = load_model(trial, env)
    
    if model is None:
        print(f'Model is None, creating new model for trial {trial.number}.')
        model = SAC(
            'MlpPolicy',
            env,
            batch_size=params['batch_size'],
            gamma=params['gamma'],
            learning_rate=params['lr'],
            policy_kwargs=params['policy_kwargs'],
            verbose=1,
            ent_coef=params['ent_coef'],
            tau=params['tau'],
            # buffer_size=params['buffer_size'],
            # learning_starts=params['learning_starts'],
            train_freq=params['train_freq']
            # gradient_steps=params['gradient_steps']
        )
        timesteps = 0

    lr_decay = trial.params['lr_reduction']
    current_lr = params['lr']

    total_time = time.time()
    
    while timesteps < total_timesteps:

        train_time = time.time()
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        train_time_stop = time.time() - train_time

        eval_time = time.time()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=200, deterministic=True)
        eval_time_stop = time.time() - eval_time
        
        timesteps += steps
        print(f'Timestep {int(timesteps)} Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} in {train_time_stop:.1f} sec with {steps/train_time_stop:.1f} fps. Eval time: {eval_time_stop:.1f} sec.')   
        
        if mean_reward > best_reward:
            print(f'New best reward: {mean_reward:.2f} > {best_reward:.2f} saving model to {best_model_path}')
            best_reward = mean_reward
            model.save(best_model_path)
        
        current_lr *= lr_decay
        model.lr_schedule = lambda _: current_lr
        print(f'New learning rate: {current_lr:.8f}')


    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')
    print(f'Total mean reward this training: {best_reward:.1f}')

    env.close()
    
    eval_env = gym.make("Hopper-v4", render_mode='rgb_array')  
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



def train_sac_baseline(env_id="Hopper-v4", total_timesteps=300000):
    # Create environment
    env = create_env(env_id, n_envs=20)
    
    # Initialize the SAC model with default hyperparameters
    model = SAC('MlpPolicy',env)

    step = 20000
    timesteps = 0
    rewards = []
    best_reward = -float('inf')
    total_time = time.time()

    while timesteps < total_timesteps:
        learn_time_start = time.time()
        model.learn(total_timesteps=step, reset_num_timesteps=False)
        timesteps += step
        learn_time_stop = time.time() - learn_time_start

        eval_time_start = time.time()
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
        eval_time_stop = time.time() - eval_time_start
        rewards.append(reward)
        mean_reward = np.mean(rewards[-3:])

        print(f'Timestep {timesteps} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {mean_reward:.1f} training time: {learn_time_stop:.1f} sec with {step/learn_time_stop:.1f} fps. Eval time: {eval_time_stop:.1f} sec.')

        if reward > best_reward:
            print(f'New best reward: {reward:.2f} > {best_reward:.2f}')
            best_reward = reward
            model.save(f"/models/beseline_sac.zip")

    print(f'Total time taken for training: {int(time.time() - total_time) / 60:.1f} min.')
    print(f'Total mean reward: {np.mean(rewards):.1f}')
    print(f'Total timesteps: {timesteps}')

    env.close()

    eval_env = gym.make("HumanoidStandup-v4", render_mode='rgb_array')  
    obs, info = eval_env.reset()
    frames = []    
    total_reward = 0.0    
    model.load(f"/models/beseline_sac.zip", env = eval_env)
    
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
    clip.write_videofile(f'/workspace/videos/hopper_sac_baseline_{total_reward:.0f}.mp4', codec='libx264')
    print(f'Saved video with total reward: {total_reward:.2f} to /workspace/videos/hopper_sac_baseline_{total_reward:.0f}.mp4')

    del model
    del env
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()

    return best_reward



def objective(trial):

    total_timesteps = int(3e5)

    params = {
        'batch_size': trial.suggest_int('batch_size', 128, 2048, log=True),
        'gamma_eps': trial.suggest_float('gamma_eps', 1e-5, 0.1, log = True),
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),        
        'tau': trial.suggest_float('tau', 1e-5, 0.1, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.1, 0.8),
        # 'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(5e5), log=True),
        # 'learning_starts': trial.suggest_int('learning_starts', 100, 10000, log=True),
        # 'train_freq': trial.suggest_int('train_freq', 1, 50),
        # 'gradient_steps': trial.suggest_int('gradient_steps', 10, 200),
        # 'actor_network_depth': trial.suggest_int('actor_network_depth', 1, 2),
        # 'critic_network_depth': trial.suggest_int('critic_network_depth', 1, 2),
        'lr_reduction': trial.suggest_float('lr_reduction', 0.9, 1.0)       
    }

    # for i in range(params['actor_network_depth']):
    #     params[f'actor_layer_{i+1}_neurons'] = trial.suggest_int(f'actor_layer_{i+1}_neurons', 32, 512, log=True)

    # for i in range(params['critic_network_depth']):
    #     params[f'critic_layer_{i+1}_neurons'] = trial.suggest_int(f'critic_layer_{i+1}_neurons', 32, 512, log=True)
    
    reward = train_and_evaluate(params, total_timesteps=total_timesteps, trial=trial)

    return reward


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

            
    # previous_study = optuna.load_study(study_name='7_01_sac_slope', storage='sqlite:///gymnasium_humanoid_walking.db')

    # last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
    #                     key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    best_reward = train_sac_baseline()
    print(f'Baseline reward: {best_reward:.1f}')

    # study_name='7_09_sac_lessparams_test8'
    study_name = 'testi'

    sampler = optuna.samplers.TPESampler(
    consider_prior=True,
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    n_startup_trials=10,
    n_ei_candidates=24,
    multivariate=True,
    group=True,
    warn_independent_sampling=False    
    )

    storage = optuna.storages.RDBStorage('sqlite:///gymnasium_hopper.db')
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_min_trials = 10),
        storage=storage,
        load_if_exists=True,
        sampler=sampler
    )

    # timesteps are not the same in the previous trials so that new trials won't be pruned with these timesteps
    # for idx, trial in enumerate(last_trials):
    #     trial.params['lr_reduction'] = 1.0
    #     print(f'Trial {trial.number} value {trial.value} user attrs {trial.user_attrs}')
    #     study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)

    study.optimize(objective, n_trials=10000, timeout= 4 * 3600)

    top_trials = sorted([t for t in study.trials if t.state != optuna.trial.TrialState.PRUNED], key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    for trial in top_trials:
        print(f"Training top trial {trial.number} with long training duration...")
        print(f'Value: {trial.value:.5f}')
        print(f'Params: {trial.params}')
    
    
        total_reward = sava_video_of_trial(trial, name = study_name)
        print(f"Video saved for trial {trial.number} with total reward: {total_reward:.2f}")
