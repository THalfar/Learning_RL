import gymnasium as gym
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, sync_envs_normalization
import os
import signal
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time


class OptunaEvalCallback(EvalCallback):
    def __init__(self, trial, factor=0.5, patience=5, threshold=0.01, min_lr=1e-6, *args, **kwargs):
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
        self.max_patience = 5
        self.all_patience = 0
        
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
                    ) from e

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

            # --- Optuna-raportointi alkaa ---
            if hasattr(self, 'trial'):
                self.trial.report(self.last_mean_reward, self.num_timesteps)

                if self.trial.should_prune():
                    print(f'Pruning trial {self.trial.number} at timestep {self.num_timesteps} with mean reward {self.last_mean_reward}')
                    continue_training = False
                    raise optuna.exceptions.TrialPruned()
            # --- Optuna-raportointi loppuu ---
            
            if mean_reward > self.best_mean_reward:

                self.wait = 0       
                self.all_patience = 0       

                self.best_mean_reward = mean_reward
                if self.verbose >= 1:
                    print("--- New best mean reward! ---")
                if self.best_model_save_path is not None:
                    print(f'Saving model to {self.best_model_save_path}')
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
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
                            print(f'Current mean reward {mean_reward} is not improving for {self.wait} steps. Reducing learning rate to {new_lr} in timestep {self.num_timesteps}')
                    self.wait = 0

            if self.max_patience <= self.all_patience:
                print(f'Max patience {self.all_patience}/{self.max_patience} reached. Stopping the study at timestep {self.num_timesteps} with mean reward {mean_reward}')
                continue_training = False
                raise optuna.exceptions.TrialPruned()
                        
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
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

            print(f'Elapsed time {(time.time() - self.start_time)/60:.1f} min.')
            self.start_time = time.time()
       
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
    
    env_name = 'HandManipulateBlockRotateParallelDense-v1'
    total_timesteps = int(15e6)

    # Suggest hyperparameters
    params = {
        'batch_size': trial.suggest_int('batch_size', 256, 512, log=True),
        'gamma_eps' : trial.suggest_float('gamma_eps', 1e-5, 1e-1, log=True),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'tau': trial.suggest_float('tau', 0.001, 0.01),
        'buffer_size' : int(7e6),
        'network_size': trial.suggest_categorical('network_size', ['tiny', 'small', 'medium', 'large', 'huge']),
        'gradient_steps': trial.suggest_int('gradient_steps', 1, 8),
        'train_freq': trial.suggest_int('train_freq', 1, 8),
        'ent_coef': trial.suggest_categorical('ent_coef', ['auto']),
    }

    policy_kwargs = {
        'net_arch': {
            'tiny': [64, 64, 64],
            'small': [128, 128, 128],
            'medium': [256, 256, 256],
            'large': [512, 512, 512],
            'huge': [1024, 1024, 1024],
        }[params['network_size']]
    }

    # Create the environment
    env = SubprocVecEnv([make_env(env_name) for _ in range(23)])
    
    # Set up the model
    model = SAC('MultiInputPolicy', env,
                batch_size=params['batch_size'],
                gamma=1 - params['gamma_eps'],
                learning_rate=params['lr'],
                tau=params['tau'],
                buffer_size=params['buffer_size'],
                policy_kwargs=policy_kwargs,
                verbose=0,
                gradient_steps=params['gradient_steps'],
                train_freq=params['train_freq'],
                ent_coef=params['ent_coef'])

    
    model_save_path = f"./models/{trial.study.study_name}_best.zip"
    eval_callback = OptunaEvalCallback(trial = trial, eval_env = env, best_model_save_path=model_save_path, log_path="./logs/",
                                 eval_freq=6e5, deterministic=True, render=False, verbose=1, n_eval_episodes=500, patience=1, factor=0.7, min_lr=1e-6)

    callback = CallbackList([eval_callback])

    start_time = time.time()

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except optuna.exceptions.TrialPruned:
        print(f'second pruning trial {trial.number}')
        optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f'Exception {e}')
        optuna.exceptions.TrialPruned()
        return float('nan')

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=500)

    # Free memory
    env.close()
    del model
    del env
    
    if eval_callback.give_best() > mean_reward:
        print(f'Best mean reward from previous {eval_callback.give_best()}')
        print(f'Mean reward from this trial {mean_reward} with time {(time.time() - start_time)/60:.1f} min.')
        return eval_callback.give_best()
    else:
        print(f'Best reward for this trial {eval_callback.give_best()} with time  {(time.time() - start_time)/60:.1f} min.')
        return mean_reward
    
if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    signal.signal(signal.SIGINT, signal_handler)
    storage = optuna.storages.RDBStorage('sqlite:///HandManipulateBlockRotateParallelDense.db')

            
    previous_study = optuna.load_study(study_name='8_13_phase2_test9', storage=storage)

    tpe = optuna.samplers.TPESampler(
    n_startup_trials=0,       
    n_ei_candidates=15,        
    multivariate=True,         
    group=True,                
    constant_liar=True,
    warn_independent_sampling=False         
    )

    study_name = 'callback_8_17_test'
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_min_trials=3),
        storage=storage,
        load_if_exists=True,
        sampler=tpe
    )

    
    last_trials = sorted([t for t in previous_study.trials if t.state != optuna.trial.TrialState.PRUNED],
                        key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]

    # for idx, trial in enumerate(last_trials):        
    #     print(f'Trial {trial.number} value {trial.value} user attrs {trial.user_attrs}')
    #     print(f'Params: {trial.params}')
    #     study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)


    study.optimize(objective, n_trials=100, timeout=400 * 3600)


