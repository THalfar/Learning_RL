# Learning_RL 
This repository showcases my training in the field of reinforcement learning (RL). It is currently in the early stages. At present, I am learning to use RL with OpenAI Gymnasium environments. I am utilizing Optuna to implement Neural Architecture Search (NAS) and Stable Baselines3 for various RL algorithms.

My goal is to integrate these training scripts into a single class once I confirm that the two-phase NAS approach is viable. This integration will streamline the process and improve efficiency.

In the future, I plan to experiment with environments that better support parallelism, such as TensorFlow Agents and other interesting and parallelizable environments like Ray RLlib and Unity ML-Agents. These environments will allow me to leverage more advanced parallel processing capabilities and explore a wider range of RL applications.


## Two phase NAS 

I am attempting to optimize Neural Architecture Search (NAS) with a two-phase approach.

**First Phase**:

- **Objective**: In this phase, I train the model with a short number of steps using Optuna.
- **Metric**: The optimization target is the slope of the linear fit to the rewards in this phase.
- **Goal**: The aim is to quickly explore the hyperparameter space and identify parameters that enable rapid learning.

**Second Phase**:

- **Objective**: I use the results from the first phase for longer training in NAS.
- **Current Metric**: I am currently experimenting with various approaches and using the moving average of the rewards from the last three training steps.
- **Goal**: To refine the hyperparameters identified in the first phase and achieve robust performance over longer training periods.
