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

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Stopping the study...')
    study.stop()
    exit(0)

# env_name = 'FetchReachDense-v2'
env_name = 'HandManipulateBlockRotateXYZDense-v1'
max_reward = -float('inf')



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
    eval_env = gym.make(env_name, render_mode='rgb_array')  
    obs, info = eval_env.reset()
    frames = []    
    total_reward = 0.0    
    model, _ = load_model(trial, eval_env, best = True)
    
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

    clip_name = f'/workspace/videos/{env_name}_{total_reward:.5f}_{trial.study.study_name}_{trial.number}.mp4'
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(clip_name, codec='libx264')
    print(f'Video saved to {clip_name}')
    
    del model    
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_reward


def train_and_evaluate(params, total_timesteps, trial=None):

    global max_reward, env_name

    env = create_env(env_name, n_envs=23)
    test_env = create_env(env_name, n_envs=23)

    model, timesteps = load_model(trial, env)
    if model is None:
        model = SAC('MultiInputPolicy',env,
                    replay_buffer_class=HerReplayBuffer,    
                    replay_buffer_kwargs = {
                    'n_sampled_goal' : params['n_sampled_goal'],
                    'goal_selection_strategy': params['goal_selection_strategy']
                    },
            batch_size=params['batch_size'],
            gamma = 1 - params['gamma_eps'],
            learning_rate=params['lr'],            
            verbose=0,
            ent_coef= f'auto_{params["ent_start"]}',
            tau=params['tau'],
            buffer_size=params['buffer_size'],
            learning_starts=3e4
            )
    timesteps = 0
    
    step = 3e5
    
    rewards = []
    current_lr = params['lr']
    best_reward = -float('inf')

    total_time = time.time()
    
    while timesteps < total_timesteps:

        try: 
            learn_time_start = time.time()        
            model.learn(total_timesteps=step, reset_num_timesteps=False)
            timesteps += step
            learn_time_stop = time.time() - learn_time_start
            
            eval_time_start = time.time()
            reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=200, deterministic=True)
            eval_time_stop = time.time() - eval_time_start        
            rewards.append(reward)
            mean_reward = np.mean(rewards[-3:])

        except Exception as e:
            print(f'Error in trial {trial.number}: {e}')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')        
            env.close()
            test_env.close()
            del model
            del env
            del test_env
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

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
            env.close()
            test_env.close()
            del model
            del env
            del test_env
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if learn_time_stop > 600:
            print(f'Trial {trial.number} exceeded limit. Pruning.')
            print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')     
            env.close()
            test_env.close()
            del model
            del env
            del test_env
            gc.collect()
            torch.cuda.empty_cache()   
            raise optuna.exceptions.TrialPruned()
        
        current_lr *= params['lr_reduction']
        model.lr_schedule = lambda _: current_lr
        print(f'New learning rate: {current_lr:.8f}')
    
        if best_reward > max_reward:
            
            print(f'\n----------------- New best reward for all trials: {best_reward:.2f} > {max_reward:.2f} ----------------- \n')
            max_reward = best_reward
            save_model(model, trial, timesteps, best = True)
            save_video_of_trial(trial)
    
    print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
    print(f'Last mean reward: {mean_reward:.1f}')
    print(f'Total timesteps: {timesteps}')
    
    env.close()
    test_env.close()
    del model
    del env
    del test_env
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_reward


def train_sac_baseline(env_id="FetchPushDense-v2", total_timesteps=1e6):
    # Create environment
    env = create_env(env_id, n_envs=22)
    
    # Initialize the SAC model with default hyperparameters
    # model = SAC('MlpPolicy',env)
    model = SAC('MultiInputPolicy',env)

    step = 2e5
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
        reward, std_reward = evaluate_policy(model, env, n_eval_episodes=40, deterministic=True)
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

    eval_env = gym.make("FetchPushDense-v2", render_mode='rgb_array')  
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
    clip.write_videofile(f'/workspace/videos/sac_baseline_{total_reward:.0f}.mp4', codec='libx264')
    print(f'Saved video with total reward: {total_reward:.2f} to /workspace/videos/sac_baseline_{total_reward:.0f}.mp4')

    del model
    del env
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()

    return best_reward



def objective(trial):

    total_timesteps = int(5e6)

    params = {
        'batch_size': trial.suggest_int('batch_size', 128, 3000, log=True),
        'gamma_eps': trial.suggest_float('gamma_eps', 1e-5, 1e-1, log = True),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),        
        'tau': trial.suggest_float('tau', 1e-6, 0.1, log=True),
        'ent_start' : trial.suggest_float('ent_start', 0.1, 0.9),
        'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(2e6), log=True),
        'lr_reduction': trial.suggest_float('lr_reduction', 0.7, 1.0),
        'n_sampled_goal': trial.suggest_float('n_sampled_goal', 0.1, 2, log = True),
        "goal_selection_strategy": trial.suggest_categorical("goal_selection_strategy", ['future', 'final', 'episode'])
    }

    # for i in range(params['actor_network_depth']):
    #     params[f'actor_layer_{i+1}_neurons'] = trial.suggest_int(f'actor_layer_{i+1}_neurons', 32, 512, log=True)

    # for i in range(params['critic_network_depth']):
    #     params[f'critic_layer_{i+1}_neurons'] = trial.suggest_int(f'critic_layer_{i+1}_neurons', 32, 512, log=True)
    
    reward = train_and_evaluate(params, total_timesteps=total_timesteps, trial=trial)

    return reward


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    signal.signal(signal.SIGINT, signal_handler)

            
    # previous_study = optuna.load_study(study_name='7_17_push_testi2', storage='sqlite:///gymnasium_reach.db')

    # last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
    #                     key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:10]

    # last_trials.reverse()

    # best_reward = train_sac_baseline()
    # print(f'Baseline reward: {best_reward:.1f}')

    study_name='7_18_hands_of_kosmos'
    
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

    storage = optuna.storages.RDBStorage('sqlite:///gymnasium_hands.db')
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
    #     trial.params['lr_reduction'] = np.random.uniform(0.7, 1.0)
    #     print(f'Trial {trial.number} value {trial.value} user attrs {trial.user_attrs}')
    #     print(f'Params: {trial.params}')
    #     study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)

    
    study.optimize(objective, n_trials=1000000, timeout= 400 * 3600)
