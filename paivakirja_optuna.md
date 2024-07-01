# Eri gymnasium Walking Optunan hakujen kuvaukset 

## 6_25_sac_walking

### Yleinen 

Käytetään Optunan standarti TPE ja parhaille 1e6 koulutus.  

### Prunetus 

Optuna maksimoi 2e5 askeleen 2e4 askelein. Prunetus tehdään joka 2e4 askeleella:
pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=42)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
Liian pitkät ajot lopetetaan:
if eval_time_stop > 180:
        print(f'Trial {trial.number} exceeded limit with {eval_time_stop}. Pruning.')

Optunan optimoinin kohde on viimeisen askeleen keskiarvo:
last_reward, last_std = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)

### Hyperparametrien avaruus

  params = {
        'batch_size': trial.suggest_int('batch_size', 32, 512, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.9999),
        'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        'total_timesteps': int(2e5),
        'tau': trial.suggest_float('tau', 1e-3, 0.1, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.5),
        'buffer_size': trial.suggest_int('buffer_size', int(1e4), int(1e6), log=True),
        'learning_starts': trial.suggest_int('learning_starts', 1000, 10000, log=True),
        'train_freq': trial.suggest_int('train_freq', 1, 100),
        'gradient_steps': trial.suggest_int('gradient_steps', 1, 100),        
        'actor_network_depth': trial.suggest_int('actor_network_depth', 1, 3),
        'critic_network_depth': trial.suggest_int('critic_network_depth', 1, 3)
    }

    for i in range(params['actor_network_depth']):
        params[f'actor_layer_{i+1}_neurons'] = trial.suggest_int(f'actor_layer_{i+1}_neurons', 32, 1024, log=True)

    for i in range(params['critic_network_depth']):
        params[f'critic_layer_{i+1}_neurons'] = trial.suggest_int(f'critic_layer_{i+1}_neurons', 32, 1024, log=True)


## 6_25_sac_slope 

Kokeillaan kulmakerrointa jotta löydetään parhaasti oppivat mallit. 