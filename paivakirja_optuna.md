# Eri Gymnasium Humanoid-v4 Optunan hakujen kuvaukset 

## 7_01_sac_slope walking_slope.py 

CmaEsSampler käyttäen optimoidaan slopea. Mukaan trialeissa user_attr:
mean3 : kolmen viimeisen askeleen reward keskiarvo 
reward : viimeisen askeleen reward
std : viimeisen askeleen std 
total_mean : kaikkien askaleiden reward keskiarvo 

Askeleita 5x1e4. 

Hyperparametrien avaruus: 


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

## 7_02_sac_walking_best_slopes 