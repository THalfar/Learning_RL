# Learning_RL 
This repository showcases my training in the field of reinforcement learning (RL). It is currently in the early stages. At present, I am learning to use RL with OpenAI Gymnasium environments. I am utilizing Optuna to implement Neural Architecture Search (NAS) and Stable Baselines3 using SAC.

My goal is to integrate these testing scripts into a single class once I confirm that the two-phase NAS approach is viable. This integration will streamline the process and improve efficiency.

## Two-phase NAS 

I am attempting to optimize Neural Architecture Search (NAS) with a two-phase approach.

**First Phase**:

- **Objective**: In this phase, I train the model with a limited number of steps using Optuna.
- **Metric**: The optimization target is the slope of the linear fit to the rewards in this phase.
- **Goal**: The aim is to quickly explore the hyperparameter space and identify parameters that enable rapid learning.

**Second Phase**:

- **Objective**: I use the results from the first phase for longer training in NAS.
- **Current Metric**: I am currently experimenting with various approaches and using the moving average of the rewards from the last three training steps.
- **Goal**: To refine the hyperparameters identified in the first phase and achieve robust performance over longer training periods.

## Current Progress

- **Class Development**: I have begun developing a class that unifies the various RL testing scripts. This class is designed to handle single-phase NAS processes, enabling streamlined experimentation with different environments and models.
- **Next Steps**: I will continue to enhance this class by integrating more optimization strategies and improving support for diverse RL environments.

## TODO
- [ ] **Clean up test scripts**: After the class implementation is complete, remove or refactor the existing test scripts, which are currently poorly named.


# Results 

## HandManipulateBlockRotateParallelDense 
[![HandManipulateBlockRotateParallelDense solution using SAC](/pictures/hand_manipulate_block.png)](https://www.youtube.com/watch?v=eGOhrHnQlEo)

Success rate calculated by running 1000 trials in test enviroment: 78.8 %


The hyperparameters for this model were optimized using a two-stage NAS process with Optuna, which identified the best hyperparameters for long training runs.
- The first stage used Optuna's QMC sampler (Quasi-Monte Carlo Sampler) with 600,000 training steps, evaluating every 300,000 steps. A linear fit slope to these intermediate steps was used as the optimization target.
- The second stage used Optuna's TPE sampler with 10,000,000 training steps, evaluating every 600,000 steps. In this phase, the 20 best hyperparameters from the first stage were used as the starting point by enqueueing these trials to the TPE sampler. The best evaluation value across all steps was considered the trial result.
- The best-found hyperparameters were used for a long training run of 120,000,000 steps.

Notably, to achieve this result, I modified the Stable Baselines EvalCallback class to reduce the learning rate whenever the model's training showed no progress. Additionally, Optuna's integration has been added to this class.

You can find the **[Best SAC model for HandManipulateBlockRotateParallelDense](/best_models/HandManipulateBlockRotateParallelDense-v1/callback_8_21_long_0_best)** in this repository.





