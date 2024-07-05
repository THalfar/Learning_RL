import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        
        return env
    return _init

if __name__ == '__main__':
        
    num_envs = 21
    env_id = 'Humanoid-v4'

    # DummyVecEnv
    env_fns = [make_env(env_id, i) for i in range(num_envs)]
    dummy_env = DummyVecEnv(env_fns)
    dummy_model = SAC('MlpPolicy', dummy_env, verbose=1)
    start_time = time.time()
    dummy_model.learn(total_timesteps=20000)
    print(f"DummyVecEnv training time: {time.time() - start_time} seconds")

    # SubprocVecEnv
    subproc_env = SubprocVecEnv(env_fns)
    subproc_model = SAC('MlpPolicy', subproc_env, verbose=1)
    start_time = time.time()
    subproc_model.learn(total_timesteps=20000)
    print(f"SubprocVecEnv training time: {time.time() - start_time} seconds")
