import gymnasium as gym
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, sync_envs_normalization
import os
import signal
import numpy as np
import time


class OptunaEvalCallback(EvalCallback):
    def __init__(self, trial, factor=0.5, patience=5, max_patience = 8, threshold=0.01, min_lr=1e-6, *args, **kwargs):
        super(OptunaEvalCallback, self).__init__(*args, **kwargs)
        self.trial = trial
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_mean_reward = -np.inf
        self.wait = 0
        self.last_check_step = 0
        self.start_time = time.time()
        self.max_patience = max_patience
        self.all_patience = 0 # Counter for max patience
        self.best_model_save_path = f'./models/{self.trial.study.study_name}'
        self.time_start = time.time()
        self.time_step = time.time()
        
    def _on_step(self) -> bool:
        
        continue_training = True

        if self.num_timesteps - self.last_check_step >= self.eval_freq:
            self.last_check_step = self.num_timesteps
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            # Optuna integration
            if hasattr(self, 'trial'):
                self.trial.report(self.last_mean_reward, self.num_timesteps)

                if self.trial.should_prune():
                    print(f'Pruning trial {self.trial.number} at timestep {self.num_timesteps} with mean reward {self.last_mean_reward}')
                    continue_training = False
                    self.best_mean_reward = self.last_mean_reward
                    raise optuna.exceptions.TrialPruned()
            
            
            if mean_reward > self.best_mean_reward:

                self.wait = 0       
                self.all_patience = 0       

                self.best_mean_reward = mean_reward
                if self.verbose >= 1:
                    print("--- New best mean reward! ---")
                    
                    if self.trial.number != 0:
                        if mean_reward > self.trial.study.best_value:
                            save_path = self.best_model_save_path + f'_{mean_reward:.4f}'
                            self.model.save(save_path)
                            print('*********************************** New study value ***********************************')
                            print(f"New best mean reward: {mean_reward:.2f}! Saving model to {save_path}")
                            print('*****************************************************************************************')
                     
                # self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
  
                
            else:

                self.wait += 1
                self.all_patience += 1
                if self.wait >= self.patience:
                    # Reduce the learning rate
                    old_lr = self.model.lr_schedule(self.model._current_progress_remaining)
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if old_lr != new_lr:
                        self.model.lr_schedule = lambda _: new_lr
                        if self.verbose > 0:
                            print(f'Current mean reward {mean_reward} is not improving for {self.wait} steps. Reducing learning rate to {new_lr} in timestep {self.num_timesteps} time taken so far {(time.time() - self.time_start)/60:.1f} min.')
                    

            if self.max_patience <= self.all_patience:
                print(f'Max patience {self.all_patience}/{self.max_patience} reached. Stopping the study at timestep {self.num_timesteps} with mean reward {mean_reward}')
                continue_training = False
                raise optuna.exceptions.TrialPruned()
                        
            if self.verbose >= 1:
                print(f"Eval num_timesteps: {self.num_timesteps}, " f"episode_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)


            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            print(f'Elapsed time {(time.time() - self.time_step)/60:.1f} min. Total run time for trial so far {(time.time() - self.time_start)/60:.1f} min.')
            self.time_step = time.time()
       
        return continue_training
    
    def give_best(self):
        return self.best_mean_reward


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Stopping the study...')
    study.stop()
    exit(0)

    
def make_env(env_id):
    def _init():
        env = gym.make(env_id)        
        env = Monitor(env)
        return env
    return _init


def objective(trial):
    
    # env_name = 'HandManipulateBlockRotateParallelDense-v1'
    env_name = 'LunarLanderContinuous-v2'
    
    total_timesteps = int(1e6)

    # Suggest hyperparameters
    params = {
        'batch_size': trial.suggest_int('batch_size', 8, 512, log=True),
        'gamma_eps' : trial.suggest_float('gamma_eps', 1e-5, 1e-1, log=True),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'tau': trial.suggest_float('tau', 0.001, 0.01),
        'buffer_size' : int(1e6),        
        'gradient_steps': trial.suggest_int('gradient_steps', 1, 4),
        'train_freq': trial.suggest_int('train_freq', 1, 4),        
        'network_size': trial.suggest_int('network_size', 8, 1024, log=True),
        'ent_coef': trial.suggest_categorical('ent_coef', ['auto']),
    }

    policy_kwargs = {
        'net_arch' : [params['network_size'], params['network_size']]
    }

    env = SubprocVecEnv([make_env(env_name) for _ in range(23)])
    
    # Set up the model
    model = SAC('MlpPolicy', env,
                batch_size=params['batch_size'],
                gamma= 1 - params['gamma_eps'],
                learning_rate=params['lr'],
                tau=params['tau'],
                buffer_size=params['buffer_size'],
                policy_kwargs=policy_kwargs,
                verbose=0,
                gradient_steps=params['gradient_steps'],
                train_freq=params['train_freq']
                )

    
    
    eval_callback = OptunaEvalCallback(trial = trial, eval_env = env, log_path="./logs/",
                                 eval_freq=1e5, deterministic=True, render=False, verbose=1, n_eval_episodes=2, patience=1, max_patience= 10, factor=0.5, min_lr=1e-7)

    callback = CallbackList([eval_callback])

    start_time = time.time()

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except optuna.exceptions.TrialPruned:
        print(f'Pruning trial {trial.number}')
        
    except Exception as e:
        print(f'Exception error: {e}')
        # optuna.exceptions.
        

        return float('-inf')

    best_reward = eval_callback.give_best()

    # Free memory
    env.close()
    del model
    del env
    del eval_callback

    print(f'Best reward for this trial {best_reward} with time  {(time.time() - start_time)/60:.1f} min.')
    
    
    return best_reward
    
if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    signal.signal(signal.SIGINT, signal_handler)
    storage = optuna.storages.RDBStorage('sqlite:///LunarLander-v2.db')

            
    # previous_study = optuna.load_study(study_name='8_13_phase2_test9', storage=storage)

    
    tpe = optuna.samplers.TPESampler(
    n_startup_trials=10,           
    multivariate=True,         
    warn_independent_sampling=False         
    )

    study_name = '8_28_lunar_nightnas'
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
        sampler=tpe
    )

    
    # last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
    #                     key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    # for idx, trial in enumerate(last_trials):        
    #     print(f'Trial {trial.number} value {trial.value} user attrs {trial.user_attrs}')
    #     print(f'Params: {trial.params}')
    #     study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)

    # study.enqueue_trial({
    # 'batch_size': 512,
    # 'gamma_eps': 0.01,
    # 'lr': 0.0006,
    # 'tau': 0.004,        
    # 'buffer_size': 13e6,
    # 'network_size': 'huge',
    # 'train_freq': 5,
    # 'gradient_steps': 5,
    # 'ent_coef': 'auto'
    # # # # })
    # importance = optuna.importance.get_param_importances(study)
    # print("Parametrien tärkeydet:")
    # for param, value in importance.items():
    #     print(f"{param}: {value:.4f}")

    study.optimize(objective, n_trials=10000, timeout=400 * 3600)


