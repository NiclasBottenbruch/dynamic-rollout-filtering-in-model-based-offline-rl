# Model-Based Offline Reinforcement Learning with Dynamic Filtering

## Project description
This code is based on the [Offline RL Kit](https://github.com/yihaosun1124/OfflineRL-Kit) by Yihao Sun et al.

For my master's thesis (which is not publicly available), I integrated the [Trajectory Transformer](https://github.com/jannerm/trajectory-transformer) by Janner as a dynamics model
in model-based offline reinforcement learning, which creates trajectories of observations, actions, and rewards jointly by maximizing the trajectory likelihood using beam search to mitigate the error accumulation of an autoregressive step-by-step process. <br>
A drawback of the Trajectory Transformer, besides its computational and memory requirements, turned out to be that the error from the beginning of the rollout trajectory is higher compared to a simple feed-forward dynamics ensemble, which in part is due to its requirement of discretization,  as an experiment showed.<br>

In this work, a simpler approach is used to increase the quality of synthetically generated data. Model rollouts, created with a feed-forward dynamics ensemble, are evaluated based on discrepancy/uncertainty measures in the ensemble and stopped once a threshold is reached. <br>
In contrast to models that learn a pessimistic MDP, such as [MOPO](https://arxiv.org/pdf/2005.13239) and [MOReL](https://arxiv.org/pdf/2005.05951), which lower the reward of transitions explicitly based on ensemble discrepancy, this work uses ensemble uncertainty measures to dynamically stop rollouts completely. The aim is to filter out a small portion of the worst data samples, which make the model-based ORL algorithm require strong regularization measures for synthetic data and limit the rollout length. The filtered synthetic data is then used in the sota mborl algorithms [COMBO](https://arxiv.org/abs/2102.08363) and [MOBILE](https://proceedings.mlr.press/v202/sun23q.html).<br>
See the adjusted rollout creation in [offlinerlkit/policy/model_based/combo.py](offlinerlkit/policy/model_based/combo.py) and [offlinerlkit/policy/model_based/combo.py](offlinerlkit/policy/model_based/combo.py).<br>

Several uncertainty measures have been added, which can be seen in the `_measure_uncertainty` function in [offlinerlkit/dynamics/ensemble_dynamics.py](offlinerlkit/dynamics/ensemble_dynamics.py) to quantify uncertainty during synthetic trajectory generation. <br>
Uncertainty measures that depend on both the discrepancy between predictions in the ensemble as well as on the predicted standard deviation are among the best suited for dynamic filtering, as rollout analyses such as [eval_discrepancy_criteria_hopper_med_combo.ipynb](eval_discrepancy_criteria_hopper_med_combo.ipynb) show.<br>

A performance comparison of COMBO and MOBILE models with and without filtering in the environments hopper-v2, halfcheetah-v2, and walker2d-v2 can be found in [model_performance_comparison.ipynb](model_performance_comparison.ipynb).
The evaluated models are provided in [models](models).<br>
Medium datasets from [D4RL](https://github.com/Farama-Foundation/d4rl.git) were used for training the models, since model-based algorithms provide the most benefit with these datasets, and there is also still the most room for improvement compared to larger datasets and/or datasets gathered by a better-performing rollout policy, as the [performance evaluation](https://github.com/yihaosun1124/OfflineRL-Kit) by Sun et al. shows.<br>

For beginners to learn about reinforcement learning, offline rl, model-based RL etc., I recommend the course [Deep Reinforcement Learning (CS 285)](https://rail.eecs.berkeley.edu/deeprlcourse/) at UC Berkeley by Sergey Levine.<br>
An introduction to offline RL is given in [this paper](https://arxiv.org/pdf/2005.01643) by Levine et al.

## Supported model-based orl algorithms
- [Conservative Offline Model-Based Policy Optimization (COMBO)](https://arxiv.org/abs/2102.08363)
- [Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning (MOBILE)](https://proceedings.mlr.press/v202/sun23q.html)


## Installation
First,
- Create a Python 3.10 environment (version is required for D4RL)
- Install MuJuCo engine, which can be downloaded from [here](https://mujoco.org/download), and install `mujoco-py` (its version depends on the version of MuJoCo engine you have installed).

Second, install D4RL:
```shell
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

Then, install the adjusted OfflineRL-Kit!
```shell
git clone https://github.com/NiclasBottenbruch/dynamic-rollout-filtering-in-model-based-offline-rl.git
cd OfflineRL-Kit
pip install -r requirements.txt
python setup.py install
```

## Train Model

To train a combo model, for example, you can run the script at [run_example/run_combo.py](https://github.com/NiclasBottenbruch/dynamic-rollout-filtering-in-model-based-offline-rl/blob/main/run_example/run_combo.py). Adjust the script with the desired parameters, such as environment, dataset, filtering criterion, threshold, etc...



