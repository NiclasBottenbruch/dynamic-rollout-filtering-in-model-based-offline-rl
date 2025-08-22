import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

from offlinerlkit.utils.evaluate import evaluate


# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        eval_create_video_freq: int = 0,
        model_save_freq: int = 25,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.eval_create_video_freq = eval_create_video_freq
        self.model_save_freq = model_save_freq
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    # create data with the model
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy() # sample from real buffer _rollout_batch_size observations shape: (batch_size, obs_dim)
                    # print("init_obss shape: ", init_obss.shape)

                    # here select model to use for rollouts
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length) # create model-based rollouts

                    # print(f"rollout_transitions['obss'].shape: {rollout_transitions['obss'].shape}")
                    # print(f"rollout_transitions['next_obss'].shape: {rollout_transitions['next_obss'].shape}")
                    # print(f"rollout_transitions['actions'].shape: {rollout_transitions['actions'].shape}")
                    # print(f"rollout_transitions['rewards'].shape: {rollout_transitions['rewards'].shape}")
                    # print(f"rollout_transitions['terminals'].shape: {rollout_transitions['terminals'].shape}")
                    # print(f"rollout_info: {rollout_info}")                    

                    self.fake_buffer.add_batch(**rollout_transitions) # add the rollouts to the fake buffer
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch} # batch with real and fake transitions - batch_size transitions
                loss = self.policy.learn(batch) # learn from the real and fake transitions
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            create_video: bool = self.eval_create_video_freq > 0 and e % self.eval_create_video_freq == 0
            eval_info = evaluate(self.eval_env,
                                 self.policy,
                                 eval_episodes=self._eval_episodes, 
                                 create_video=create_video, 
                                 video_dir=self.logger.video_dir, 
                                 vid_file_name=f"eval_{e}_epochs.mp4")

            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_reward_p20 = np.percentile(eval_info["eval/episode_reward"], 20)
            ep_reward_p80 = np.percentile(eval_info["eval/episode_reward"], 80)
            eval_eisodes = len(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

            try:
                norm_score_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_score_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100

                self.logger.logkv("eval/normalized_score_episode_reward", norm_score_ep_rew_mean)
                self.logger.logkv("eval/normalized_score_episode_reward_std", norm_score_ep_rew_std)
            except AttributeError:
                # if the environment does not have get_normalized_score method use the mean reward (which is not normalized)
                norm_score_ep_rew_mean = ep_reward_mean
                
            last_10_performance.append(norm_score_ep_rew_mean)

            self.logger.logkv("eval/episode_reward_mean", ep_reward_mean)
            self.logger.logkv("eval/episode_reward_std", ep_reward_std)
            self.logger.logkv("eval/episode_reward_p20", ep_reward_p20)
            self.logger.logkv("eval/episode_reward_p80", ep_reward_p80)
            self.logger.logkv("eval/num_eval_episodes", eval_eisodes)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_state_dict.pth"))

            if self.model_save_freq > 0 and e % self.model_save_freq == 0:
                rew = np.round(norm_score_ep_rew_mean, 2)
                torch.save(self.policy, os.path.join(self.logger.model_dir, f"policy_model_{e}_epochs_{rew}_reward.pt"))
                torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, f"policy_state_dict_{e}_epochs_{rew}_reward.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        # save final model
        rew = np.round(norm_score_ep_rew_mean, 2)
        torch.save(self.policy, os.path.join(self.logger.model_dir, f"policy_model_final_{self._epoch}_epochs.pt"))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, f"policy_state_dict_final_{self._epoch}_epochs.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}
