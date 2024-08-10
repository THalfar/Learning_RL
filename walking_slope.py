import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import moviepy.editor as mpy
import tensorflow as tf
import time
import gc
import torch
import os
import signal


from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv as FetchPushEnv
import inspect


# env_name = 'FetchReachDense-v2'
# env_name = 'FetchPickAndPlaceDense-v2'
# env_name = 'FetchSlideDense-v2'
# env_name = 'HandManipulateBlockFull-v1'
# env_name = 'FetchPushDense-v2'
# # env_name = 'FetchReach-v2'
env_name = 'HandManipulateBlockRotateParallelDense-v1'




max_reward = -float('inf')

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Stopping the study...')
    study.stop()
    exit(0)



def create_env(env_id, n_envs=1):

    def make_env():
        return gym.make(env_id)

    envs =  SubprocVecEnv([make_env] * n_envs)
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)
    return envs

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


def save_model(model, trial, timesteps, best = False):
    
    if not best:
        model_path = f"./models/{trial.study.study_name}_{trial.number}.zip"
    else:
        model_path = f"./models/{trial.study.study_name}_best.zip"
        
    print(f"Saving model for trial to {model_path}")

    if not best:
        trial.set_user_attr('model_path', model_path)
        trial.set_user_attr('timesteps', timesteps)

    model.save(model_path)
    

def load_model(trial, env, best = False):
    
    if not best:
        model_path = trial.user_attrs.get('model_path', f"./models/{trial.study.study_name}_{trial.number}.zip")
    else:
        model_path = f"./models/{trial.study.study_name}_best.zip"

    if os.path.exists(model_path):
        print(f"Loading existing model for trial {trial.number} from {model_path}")
        model = SAC.load(model_path, env)
        timesteps = trial.user_attrs.get('timesteps', 0)
        return model, timesteps
    
    return None, 0


def save_video_of_trial(trial):

    global env_name
    env =  gym.make(env_name, render_mode='rgb_array')    

    obs, info = env.reset()
    frames = []    
    total_reward = 0.0    
    model, _ = load_model(trial, env, best = True)

    max_obs_values = {key: -np.inf for key in obs.keys()}

    for _ in range(50):

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        total_reward += reward

        for key in obs.keys():
            max_obs_values[key] = np.maximum(max_obs_values[key], obs[key])

        
        if done or trunc:
            break

    env.close()

    # for key, value in max_obs_values.items():
    #     print(f'Maximum observation values for {key}: {value}')

    clip_name = f'/workspace/videos/{env_name}_{total_reward:.5f}_{trial.study.study_name}_{trial.number}.mp4'
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(clip_name, codec='libx264')
    print(f'Video saved to {clip_name}')
    
    del model    
    del env
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_reward

def train_and_evaluate(params, total_timesteps, trial=None):

    global env_name, max_reward
    
    env = create_env(env_name, n_envs=18)
    test_env = create_env(env_name, n_envs=18)

    if params['network_size'] == 'tiny':
        policy_kwargs = {'net_arch' : [64, 64, 64]}
    elif params['network_size'] == 'small':
        policy_kwargs = {'net_arch' : [128, 128, 128]}
    elif params['network_size'] == 'medium':
        policy_kwargs = {'net_arch' : [256, 256, 256]}
    elif params['network_size'] == 'large':
        policy_kwargs = {'net_arch' : [512, 512, 512]}
    elif params['network_size'] == 'huge':
        policy_kwargs = {'net_arch' : [1024, 1024, 1024]}

    
    model, timesteps = load_model(trial, env)
    if model is None:
        model = SAC('MultiInputPolicy',env,
                    replay_buffer_class=HerReplayBuffer,    
                    replay_buffer_kwargs = {
                    'n_sampled_goal' : params['n_sampled_goal'],
                    'goal_selection_strategy': params['goal_selection_strategy']
                    },
            policy_kwargs = policy_kwargs,
            batch_size=params['batch_size'],
            gamma = 1 - params['gamma_eps'],
            learning_rate=params['lr'],            
            verbose=0,
            ent_coef= f'auto_{params["ent_start"]}',
            tau=params['tau'],
            buffer_size=params['buffer_size'],
            learning_starts=1e4
            )
    timesteps = 0
    
    rewards = []
    total_time = time.time()
    steps = 1e5
    reward_max = -float('inf')
    
    while True:
        try:
            learn_time_start = time.time()        
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
            timesteps += steps
            learn_time_stop = time.time() - learn_time_start

           
            eval_time_start = time.time()
            reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=200, deterministic=True)
            eval_time_stop = time.time() - eval_time_start                    
            rewards.append(reward)
            running_mean = np.mean(rewards[-3:])
            total_mean = np.mean(rewards)

            if len(rewards) < 2:
                slope = reward
            else:
                slope, _ = np.polyfit(range(len(rewards)), rewards, 1)

            if reward > reward_max:
                reward_max = reward
                
            print(f'Timestep {timesteps} slope: {slope:.2f} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {running_mean:.1f} total mean: {total_mean:.1f} training time: {int(learn_time_stop)} sec with {steps/learn_time_stop:.1f} fps. . Eval time: {int(eval_time_stop)} sec.')        

            trial.report(slope, timesteps)

           
        

        except Exception as e:
            print(f'Error in trial {trial.number}: {e}')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            trial.set_user_attr('error', str(e))
            trial.report(float('nan'), timesteps) 
            env.close()
            del model
            del env
            del test_env
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
    
        if reward > max_reward:
            print(f'--------   New max reward: {reward:.2f} > {max_reward:.2f}. Saving model. --------')
            max_reward = reward
            save_model(model, trial, timesteps, best = True)            
            save_video_of_trial(trial)
        
        if trial.should_prune():
            
            print(f'Pruner want to prune trial {trial.number} at timestep {timesteps} with value {slope}.')

            if slope < 5:
                env.close()
                del model
                del env
                del test_env
                gc.collect()
                torch.cuda.empty_cache()
                print(f'Trial {trial.number} pruned at timestep {timesteps}.')                
                print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
                raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 480:
            env.close()
            del model
            del env
            del test_env
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            raise optuna.exceptions.TrialPruned()
        
        if timesteps >= total_timesteps:
            break
    
    eval_time_start = time.time()
    reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=200, deterministic=True)
    eval_time_stop = time.time() - eval_time_start

    trial.set_user_attr('mean3', running_mean) 
    trial.set_user_attr('reward', reward)
    trial.set_user_attr('std', std_reward)
    trial.set_user_attr('total_mean', total_mean)
    trial.set_user_attr('slope', slope)

    save_model(model, trial, timesteps)    

    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
    print(f'Total mean reward this training: {np.mean(rewards):.1f}')
    print(f'Total timesteps: {timesteps}')
    
    env.close()
    del model
    del env
    del test_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return slope


def objective(trial):
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    
    total_timesteps = int(3e5)

    params = {
        'batch_size': trial.suggest_int('batch_size', 32, 4200, log=True),
        'gamma_eps' : trial.suggest_float('gamma_eps', 1e-8, 1e-1, log=True),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),        
        'tau': trial.suggest_float('tau', 1e-6, 0.1, log=True),
        'ent_start' : trial.suggest_float('ent_start', 0.1, 0.9),
        'buffer_size': trial.suggest_int('buffer_size', int(1e5), int(1e6), log=True),
        'n_sampled_goal': trial.suggest_float('n_sampled_goal', 1, 4),
        "goal_selection_strategy": trial.suggest_categorical("goal_selection_strategy", ['future', 'final', 'episode']),
        'network_size': trial.suggest_categorical('network_size', ['tiny', 'small', 'medium', 'large', 'huge'])
        }

    
    reward = train_and_evaluate(params, total_timesteps=total_timesteps, trial=trial)

    return reward

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)
    
    storage = optuna.storages.RDBStorage('sqlite:///HandManipulateBlockRotateParallelDense.db')

    qmc = optuna.samplers.QMCSampler(warn_independent_sampling=False)

    tpe = optuna.samplers.TPESampler(
    consider_prior=True,    
    consider_magic_clip=True,
    consider_endpoints=False,
    n_startup_trials=0,    
    multivariate=True,
    group=True,
    warn_independent_sampling=False    
    )

    es_sampler = optuna.samplers.CmaEsSampler(warn_independent_sampling=False)

    # tpe = optuna.samplers.TPESampler(warn_independent_sampling = False, n_startup_trials=0)

    study = optuna.create_study(
        study_name='8_9_slope_6',        
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_min_trials=10),
        storage=storage,
        load_if_exists=True,
        sampler = qmc
    )

    # study.optimize(objective, n_trials=10)
    # study.sampler = es_sampler
    study.optimize(objective, n_trials=10000000, timeout= 24 * 3600)
