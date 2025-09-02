import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

from offlinerlkit.utils.evaluate import evaluate


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        eval_create_video_freq: int = 0, # if eval_create_video_freq = 0, no videos will be created
        model_save_freq: int = 25,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
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
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
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
            norm_score_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_score_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_score_ep_rew_mean)
            self.logger.logkv("eval/normalized_score_episode_reward", norm_score_ep_rew_mean)
            self.logger.logkv("eval/normalized_score_episode_reward_std", norm_score_ep_rew_std)
            self.logger.logkv("eval/episode_reward_mean", ep_reward_mean)
            self.logger.logkv("eval/episode_reward_std", ep_reward_std)
            self.logger.logkv("eval/episode_reward_p20", ep_reward_p20)
            self.logger.logkv("eval/episode_reward_p80", ep_reward_p80)
            self.logger.logkv("eval/#eval_episodes", eval_eisodes)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_state_dict.pth"))

            if self.model_save_freq > 0 and e % self.model_save_freq == 0:
                rew = np.round(norm_score_ep_rew_mean, 2)
                torch.save(self.policy, os.path.join(self.logger.model_dir, f"policy_model_{e}_epochs_{rew}_reward.pth"))
                torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, f"policy_state_dict_{e}_epochs_{rew}_reward.pth"))


        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        # save final model
        rew = np.round(norm_score_ep_rew_mean, 2)
        torch.save(self.policy, os.path.join(self.logger.model_dir, f"policy_model_final_{self._epoch}_epochs.pth"))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, f"policy_state_dict_final_{self._epoch}_epochs.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}
    