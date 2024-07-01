import optuna
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def create_env():
    env = gym.make('CartPole-v1')
    env = Monitor(env)
    return env

def objective(trial):
    env = DummyVecEnv([create_env])

    n_steps = trial.suggest_int('n_steps', 128, 2048, step=128)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.9999)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

    model = PPO('MlpPolicy', env, n_steps=n_steps, gamma=gamma, learning_rate=lr, verbose=0)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    episode_rewards = []
    episode_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            episode_rewards.append(episode_reward)
            obs = env.reset()
            episode_reward = 0

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Render the best model
env = DummyVecEnv([create_env])

model = PPO('MlpPolicy', env, **study.best_params)
model.learn(total_timesteps=10000)

obs = env.reset()
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
