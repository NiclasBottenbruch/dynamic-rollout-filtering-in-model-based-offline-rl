from offlinerlkit.utils.video_wrapper import VideoWrapper
from typing import Optional, Dict, List
import os

def evaluate(eval_env,
             policy, eval_episodes:int = 10, 
             create_video:bool = False, 
             video_dir:str = None, 
             vid_file_name: str = "video.mp4") -> Dict[str, List[float]]:
        """
        Evaluate the policy on the environment
        Args:
            eval_env (gym.Env): Environment to evaluate the policy on
            policy (BasePolicy): Policy to evaluate
            eval_episodes (int): Number of episodes to evaluate the policy
            create_video (bool): Whether to create a video of the evaluation
            video_dir (str): Directory to save the video
            vid_file_name (str): Name of the video file
        
        Returns:
            Dict[str, List[float]]: Dictionary containing the evaluation metrics
        """

        env = eval_env
        if create_video:
            assert video_dir is not None, "video_dir must be provided if create_video is True"
            save_path = os.path.join(video_dir, vid_file_name)
            env = VideoWrapper(env, save_path)

        policy.eval()
        obs = env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < eval_episodes:
            action = policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
