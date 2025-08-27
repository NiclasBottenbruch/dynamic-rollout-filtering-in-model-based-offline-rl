import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        monitor_uncertainty_stats: list = ["aleatoric", "pairwise-diff", "pairwise-diff_with_std", "ensemble_std", "dimensionwise_diff_with_std"]
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self._monitor_uncertainty_stats = monitor_uncertainty_stats

    def _measure_uncertainty(self, mean_predicted: np.ndarray, std_predicted: np.ndarray, uncertainty_mode: str) -> np.ndarray:
        """Measure uncertainty or discrepancy based on predicted mean and std for the next observation (and reward) from ensemble dynamics model
        Args:
            mean_predicted: Mean predicted by ensemble dynamics model shape: (num_ensemble, batch_size, obs_dim + I[_with_reward])
            std_predicted: Std predicted by ensemble dynamics model shape: (num_ensemble, batch_size, obs_dim + I[_with_reward])
            uncertainty_mode: Mode to measure uncertainty or discrepancy
                "aleatoric": maximum uncertainty across all models in ensemble based on predicted std
                "pairwise-diff": maximum deviation from mean across all models in ensemble
                "pairwise-diff_with_std": maximum of deviation from mean across all models in ensemble plus predicted std
                "ensemble_std": std of predicted next observations across all models in ensemble
                "dimensionwise_diff_with_std": norm of dimension-wise difference between upper and lower bound of predicted next observations across all models in ensemble
        Returns:
            uncertainty_measure: Uncertainty or discrepancy stmeasuretistics shape: (batch_size,)
        """

        if uncertainty_mode == "aleatoric":
            uncertainty_measure = np.amax(np.linalg.norm(std_predicted, axis=2), axis=0) # [batch_size] - maximum uncertainty across all models in ensemble based on predicted std
        elif uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean_predicted[..., :-1] # [num_ensemble, batch_size, obs_dim] - mean of predicted next observations
            next_obs_mean = np.mean(next_obses_mean, axis=0) # [batch_size, obs_dim] - mean of all models in ensemble
            diff = next_obses_mean - next_obs_mean
            uncertainty_measure = np.amax(np.linalg.norm(diff, axis=2), axis=0) # [batch_size] - maximum deviation from mean across all models in ensemble
        elif uncertainty_mode == "pairwise-diff_with_std":
            next_obses_mean = mean_predicted[..., :-1] # [num_ensemble, batch_size, obs_dim] - mean of predicted next observations
            next_obs_mean = np.mean(next_obses_mean, axis=0) # [batch_size, obs_dim] - mean of all models in ensemble
            next_obses_std = std_predicted[..., :-1] # [num_ensemble, batch_size, obs_dim] - std of predicted next observations
            diff_with_std = np.abs(next_obses_mean - next_obs_mean) + next_obses_std
            uncertainty_measure = np.amax(np.linalg.norm(diff_with_std, axis=2), axis=0) # [batch_size] 
        elif uncertainty_mode == "ensemble_std":
            next_obses_mean = mean_predicted[..., :-1]
            uncertainty_measure = np.sqrt(next_obses_mean.var(0).mean(1)) # [batch_size] - std of predicted next observations across all models in ensemble
        elif uncertainty_mode == "dimensionwise_diff_with_std":
            next_obses_mean = mean_predicted[..., :-1]
            next_obses_std = std_predicted[..., :-1]
            lower = np.amin(next_obses_mean - next_obses_std, axis=0) # [batch_size, obs_dim] - lower bound of predicted next observations across all models in ensemble
            upper = np.amax(next_obses_mean + next_obses_std, axis=0) # [batch_size, obs_dim] - upper bound of predicted next observations across all models in ensemble
            uncertainty_measure = np.linalg.norm(upper - lower, axis=1) # [batch_size]
        else:
            raise ValueError(f"Unrecognized uncertainty mode: {uncertainty_mode}")
        
        assert uncertainty_measure.shape == (mean_predicted.shape[1],) # shape should be (batch_size,)
        return uncertainty_measure

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        measure_uncertainty: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """imagine single forward step
        Args:
            obs: observation of the environment shape: (batch_size, obs_dim)
            action: action to take shape: (batch_size, action_dim)
        Returns:
            next_obs: next observation of the environment shape: (batch_size, obs_dim)
            reward: reward received shape: (batch_size, 1)
            terminal: whether the episode is done shape: (batch_size, 1)
            info: additional information, e.g. penalty for uncertainty
        """
        
        obs_act = np.concatenate([obs, action], axis=-1) # (batch_size, obs_dim + action_dim)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act) # mean (delta obs and abs reward) and logvar shape: (num_ensemble, batch_size, obs_dim + I[_with_reward])
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs # add current observation to mean (delta obs) to get next observation
        std = np.exp(logvar / 2) # = np.sqrt(np.exp(logvar))        

        # sample next observation and reward - assuming gaussian distribution with zero covariance between dimensions
        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32) # [num_ensemble, batch_size, obs_dim + I[_with_reward]]

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size) # [batch_size] - random indices of elite models for each sample
        samples = ensemble_samples[model_idxs, np.arange(batch_size)] # [batch_size, obs_dim + I[_with_reward]] - take sample from the chosen model 
        
        next_obs = samples[..., :-1] # [batch_size, obs_dim] - next observation
        reward = samples[..., -1:] # [batch_size, 1] - reward
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        
        if self._penalty_coef: # penalize reward based on uncertainty
            penalty = self._measure_uncertainty(mean, std, self._uncertainty_mode) # [batch_size,] - uncertainty measure
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty # [batch_size, 1] - penalized reward
            info["penalty"] = penalty

        if measure_uncertainty:
            # log additional uncertainty statistics to monitor
            uncertainty_measures = {}
            for mode in self._monitor_uncertainty_stats:
                uncertainty_measures[mode] = self._measure_uncertainty(mean, std, mode) # [batch_size,]
            info["uncertainty_measures"] = uncertainty_measures
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
