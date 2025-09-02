import imageio
import numpy as np
import gym

class VideoWrapper(gym.Wrapper):
    """
    Wrapper for creating videos of runs in the environment.
    """

    def __init__(self, env, save_path=None, file_format=".mp4"):
        super().__init__(env)
        self.file_format = file_format
        self.save_path = save_path or f"tmp/{self.env.spec.id}_{np.random.randint(1e5, 1e6)}{self.file_format}"

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.img_lst.append(self.render(mode="rgb_array"))
        if done:
            self.save(self.save_path)
        return obs, reward, done, info

    def reset(self):
        self.img_lst = []
        obs = self.env.reset()
        self.img_lst.append(self.env.render(mode="rgb_array"))
        return obs

    def save(self, save_path, fps=29):
        imageio.mimsave(save_path, np.array(self.img_lst), fps=fps)