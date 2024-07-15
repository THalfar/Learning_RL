import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import moviepy.editor as mpy
import tensorflow as tf
import time
import gc
import torch
import os

class OptunaSAC:
    def __init__(self, env_id="Hopper-v4", total_timesteps=300000, optimization_time= 3600 * 2, verbose = 1, study_name="sac_study", step = 20000):
         
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.n_envs = 22
        self.n_eval_episodes = 50
        self.total_timesteps = total_timesteps
        self.step = step
        self.optimization_time = optimization_time
        self.verbose = verbose
        self.study_name = study_name
        self.max_step_time = 360
        self.sql_name = 'sqlite:///class_test.db'
        
        self.param_choose = {
            'batch_size': True,
            'gamma_eps': True,
            'lr': True,
            'tau': True,
            'ent_start': True,
            'lr_reduction': True,            
             }

        self.best_reward_all = -np.inf


    def suggest_params(self, trial):

        assert any(self.param_choose.values()), "At least one parameter must be chosen."
        params = {}

        if self.param_choose.get('batch_size', False):
            params['batch_size'] = trial.suggest_int('batch_size', 64, 3000, log=True)
        if self.param_choose.get('gamma_eps', False):
            params['gamma_eps'] = trial.suggest_float('gamma_eps', 1e-5, 1e-2, log=True)
        if self.param_choose.get('lr', False):
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        if self.param_choose.get('tau', False):
            params['tau'] = trial.suggest_float('tau', 1e-5, 0.1, log=True)
        if self.param_choose.get('ent_start', False):
            params['ent_start'] = trial.suggest_float('ent_start', 0.1, 0.9)
        if self.param_choose.get('lr_reduction', False):
            params['lr_reduction'] = trial.suggest_float('lr_reduction', 0.5, 1.0)   

        return params 


    def create_env(self):
        def make_env():
            return gym.make(self.env_id)
        return SubprocVecEnv([make_env for _ in range(self.n_envs)])
    
        
    def train_and_evaluate(self, params, trial):

        sac_kwargs = {}
        
        env = self.create_env()
        # print(f'Trial {trial.number} started with params: {params}.')

        if params.get('batch_size', False):
            sac_kwargs['batch_size'] = params['batch_size']
        if params.get('gamma_eps', False):
            sac_kwargs['gamma'] = 1 - params['gamma_eps']
        if params.get('lr', False):
            sac_kwargs['learning_rate'] = params['lr']
        if params.get('tau', False):
            sac_kwargs['tau'] = params['tau']
        if params.get('ent_start', False):
            sac_kwargs['ent_coef'] = f'auto_{params["ent_start"]}'

        rewards = []
        best_reward = -float('inf')
        total_time = time.time()
        timestep = 0

        model = SAC('MlpPolicy', env, verbose=0, **sac_kwargs)
        model_path = f'models/{self.study_name}_{self.env_id}_{trial.number}.zip'

        if params.get('lr_reduction', False):
            current_lr = params['lr']
        
        while timestep < self.total_timesteps:
            
            try:  
                learn_time_start = time.time()      
                model.learn(total_timesteps=self.step, reset_num_timesteps=False) 
                timestep += self.step
                learn_time_stop = time.time() - learn_time_start
                eval_time_start = time.time()
                reward, std_reward = evaluate_policy(model, env, n_eval_episodes= self.n_eval_episodes, deterministic=True)
                eval_time_stop = time.time() - eval_time_start        
                rewards.append(reward)
                mean_reward = np.mean(rewards[-3:])
            except Exception as e:
                print(f'Error in trial {trial.number} timestep {timestep}.')
                print(e)
                env.close()
                del model
                del env
                gc.collect()
                torch.cuda.empty_cache()
                return -np.inf

            if self.verbose > 0:
                model.get_parameters()
                print(f'Trial {trial.number} timestep {timestep} reward: {reward:.2f} +/- {std_reward:.2f} running mean: {mean_reward:.1f} training time: {learn_time_stop:.1f} sec with {self.step/learn_time_stop:.1f} fps. Eval time: {eval_time_stop:.1f} sec.')        

            trial.report(mean_reward, timestep)

            if reward > best_reward:
                if self.verbose > 0:
                    print(f'New best reward: {reward:.2f} > {best_reward:.2f}')
                best_reward = reward
                trial.user_attrs['best_reward'] = best_reward
                model.save(model_path)
                
            
            if trial.should_prune():
                if self.verbose > 0:
                    print(f'Trial {trial.number} pruned at timestep {timestep}.')                
                    print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')     
                env.close()
                del model
                del env
                gc.collect()
                torch.cuda.empty_cache()    
                raise optuna.exceptions.TrialPruned()
            
            if learn_time_stop > self.max_step_time:
                if self.verbose > 0:
                    print(f'Trial {trial.number} exceeded limit. Pruning.')
                    print(f'Total time taken: {int(time.time() - total_time)/ 60:.1f} min.')   
                env.close()
                del model
                del env
                gc.collect()
                torch.cuda.empty_cache()     
                raise optuna.exceptions.TrialPruned()
                        
            if params.get('lr_reduction', False):
                current_lr *= params['lr_reduction']
                model.lr_schedule = lambda _: current_lr
                
          
        
        
        if self.verbose > 0:
            print(f'Total time taken for trial {trial.number}: {int(time.time() - total_time)/ 60:.1f} min.')    
            print(f'Last mean reward: {mean_reward:.1f}')
            print(f'Total timesteps: {timestep}')

        if best_reward > self.best_reward_all:
            self.best_reward_all = best_reward
            if self.verbose > 0:
                print(f'### New best reward for this run ###: {best_reward:.1f}')            
            self.save_video_of_model(model_path, trial.number)
        
        env.close()
        del model
        del env
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_reward


    def save_video_of_model(self, model_path = None, trial_num = None) :
        assert model_path is not None, "Model must be provided."
        assert trial_num is not None, "Trial number must be provided."
        
        eval_env = gym.make(self.env_id, render_mode='rgb_array')  
        model = SAC.load(model_path, eval_env)

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

        video_path = f'videos/{self.env_id}_{total_reward:.0f}_{self.study_name}_{trial_num}_.mp4'

        clip = mpy.ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path, codec='libx264')

        if self.verbose > 0:
            print(f'Video saved to {video_path} with reward {total_reward}.')
        
        del model    
        del eval_env
        gc.collect()
        torch.cuda.empty_cache()

    
    def objective(self, trial):

        params = self.suggest_params(trial)
        return self.train_and_evaluate(params, trial)
    

    def optimize(self):

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

        storage = optuna.storages.RDBStorage(self.sql_name)
        study = optuna.create_study(
            study_name= self.study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials = 10, n_min_trials = 10),
            storage=storage,
            load_if_exists=True,
            sampler=sampler
        )

        study.optimize(self.objective, timeout=self.optimization_time)


if __name__ == '__main__':
    
    optuna_sac = OptunaSAC(env_id='HumanoidStandup-v4',study_name="7_13_standup", optimization_time=3600*84, step=200000, total_timesteps=2000000, verbose=1)
    optuna_sac.optimize()
    


    

            
        


        
