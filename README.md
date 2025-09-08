# Model-Based Offline Reinforcement Learning with Dynamic Filtering

This code is based on the [Offline RL Kit](https://github.com/yihaosun1124/OfflineRL-Kit) by Yihao Sun et al.

For my master's thesis (which is not publicly available), I integrated the [Trajectory Transformer](https://github.com/jannerm/trajectory-transformer) by Janner as a dynamics model
in model-based offline reinforcement learning which creates trajectories of observations, actions and rewards jointly by maximizing the trajectory likelihood using beam search to mitigate the error accumulation of an autoregressive step-by-step process. <br>
A drawback of the Trajectory Transformer, beside it's computational and memory requirements, turned out to be that the error from the beginning of the rollout trajectory is higher compared to a simple feed-forward dynamics ensemble, which in part is due to it's requrement of discretization as an experiment showed.<br>

In this work, a simpler approach is used to increase the quality of synthetically generated data. Model rollouts, created with a feed-forward dynamics ensemble, are evaluated based on discrepancy- / uncertainty measures in the ensemble and stopped once a threshold is reached. <br>
In contrast to models that learn a pessimistic MDP such as [MOPO](https://arxiv.org/pdf/2005.13239) and [MOReL](https://arxiv.org/pdf/2005.05951) which lower the reward of transitions explicitly based on ensemble discrepancy, this work uses ensemble uncertainty measures to dynamically stop rollouts completely. The aim is to filter out a small portion of the worst data samples, which make the model-based ORL algorithm require strong regularization measures for synthetic data and limit the rollout length. The filtered synthetic data is then used in the sota mborl algorithms [COMBO](https://arxiv.org/abs/2102.08363) and [MOBILE](https://proceedings.mlr.press/v202/sun23q.html).<br>
See the adjusted rollout creation in [offlinerlkit/policy/model_based/combo.py](offlinerlkit/policy/model_based/combo.py) and [offlinerlkit/policy/model_based/combo.py](offlinerlkit/policy/model_based/combo.py).


Several uncertailty measures have been added, which can bee seen in the `_measure_uncertainty` function in [offlinerl/algo/modelbase/ensemble_dynamics.py](`offlinerl/algo/modelbase/ensemble_dynamics.py`) to quantify uncertainty during synthetic trajectory generation. <br>
Uncertainty measures that depend on both the the discrepancy between predictions in the ensemble as well as on the predicted standard deviation are among the best suited for dynamic filtering as rollout analysis such as [eval_discrepancy_criteria_hopper_med_combo.ipynb](eval_discrepancy_criteria_hopper_med_combo.ipynb) show.

## Supported algorithms
- Model-free - as Benchmark
    - [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779)
- Model-based - where dynamic rollout filtering is applicable
    - [Conservative Offline Model-Based Policy Optimization (COMBO)](https://arxiv.org/abs/2102.08363)
    - [Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning (MOBILE)](https://proceedings.mlr.press/v202/sun23q.html)

## Benchmark Results (4 seeds) (measured by Yihao Sun et al.)

|                              | CQL       | TD3+BC    | EDAC      | IQL       | MOPO      | RAMBO     | COMBO     | MOBILE     |
| ---------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| halfcheetah-medium-v2        | 49.4±0.2  | 48.2±0.5  | 66.4±1.1  | 47.4±0.5  | 72.4±4.2  | 78.7±1.1  | 71.9±8.5  | 75.8±0.8  |
| hopper-medium-v2             | 59.1±4.1  | 60.8±3.4  | 101.8±0.2 | 65.7±8.1  | 62.8±38.1 | 82.1±38.0 | 84.7±9.3  | 103.6±1.0  |
| walker2d-medium-v2           | 83.6±0.5  | 84.4±2.1  | 93.3±0.8  | 81.1±2.6  | 84.1±3.2  | 86.1±1.0  | 83.9±2.0  | 88.3±2.5  |

## Results
|                              | CQL       | COMBO     | COMBO + FILTERING | MOBILE    | MOBILE + FILTERING   |
| ---------------------------- | --------- | --------- | ----------------- | --------- | -------------------- |
| halfcheetah-medium-v2        | 49.4±0.2  | 71.9±8.5  | 1                 |75.8±0.8   |  2                   |
| hopper-medium-v2             | 59.1±4.1  | 84.7±9.3  | 1                 |103.6±1.0  | 2                    |
| walker2d-medium-v2           | 83.6±0.5  | 83.9±2.0  | 1                 |88.3±2.5   | 2                    |

## Installation
First,
- Create a Python 3.10 environment (version is required for d4rl)
- install MuJuCo engine, which can be download from [here](https://mujoco.org/download), and install `mujoco-py` (its version depends on the version of MuJoCo engine you have installed).

Second, install D4RL:
```shell
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

Then, install the OfflineRL-Kit!
```shell
git clone https://github.com/NiclasBottenbruch/dynamic-rollout-filtering-in-model-based-offline-rl.git
cd OfflineRL-Kit
pip install -r requirements.txt
python setup.py install
```

## Train Model

To train a combo model for example, you can run the script at [run_example/run_combo.py](https://github.com/NiclasBottenbruch/dynamic-rollout-filtering-in-model-based-offline-rl/blob/main/run_example/run_combo.py). Adjust the script with the desired parameters such as environment, dataset, filtering criterion and threshold, etc...



