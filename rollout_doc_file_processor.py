import gym
import numpy as np
import json
import os
from typing import Union


class custom_wrapped_env(gym.Wrapper):
    def __init__(self, env):
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

    def get_obs(self):
        return self.env.unwrapped._get_obs()
    
    def get_qpos(self):
        return self.env.sim.data.qpos.ravel().copy()
    def get_qvel(self):
        return self.env.sim.data.qvel.ravel().copy()
    
    def set_state(self, obs, qpos_0=0.0):
        # qpos_0 is the x position of the hopper - it got removed from the observation and is insignificant for the dynamics
        qpos = np.concatenate([[qpos_0], obs[:self.env.sim.model.nq-1]], axis=0)
        qvel = obs[self.env.sim.model.nq-1:]
        self.env.set_state(qpos, qvel)
        self.env.sim.forward()

def encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def add_model_error_to_rollout_doc(rollout_doc_path: Union[str, list], env_name:str, verbose:bool=False, force:bool=False):
    """
        Adds model error information to the rollout document.
            Args:
                rollout_doc_path: The path to the json rollout document or a list of paths.
                env_name: The name of the environment.
                verbose: Whether to print verbose output.
                force: Whether to force reprocessing of the document.
            Returns:
                The updated rollout document (json) if single document is processed. None if a list of documents is processed.
    """
    if isinstance(rollout_doc_path, list):
        for path in rollout_doc_path:
            if verbose:
                print(f"Processing rollout document {path}")
            add_model_error_to_rollout_doc(path, env_name, verbose, force)
        return

    doc = json.load(open(rollout_doc_path, 'r'))

    if "next_obss_real" not in doc or "model_error_l2" not in doc or force:
        # make lists to numpy arrays
        for k,v in doc.items():
            doc[k] = np.array(v)

        N = doc[list(doc.keys())[0]].shape[0]
        if verbose:
            print(f"{N} transitions loaded from {rollout_doc_path}")

        env = custom_wrapped_env(env_name)
        env.reset()

        next_obs_real = []
        i = 0
        for obs, act, next_obs_pred in zip(doc['obss'], doc['actions'], doc['next_obss_predicted']):
            if i % 1000 == 0 and verbose:
                print(f"{i} / {N}")
            i += 1
            env.set_state(obs)
            nor = env.step(act)[0]
            next_obs_real.append(nor)
        
        next_obs_real = np.array(next_obs_real)
        model_error_l2 = np.linalg.norm(next_obs_real - doc['next_obss_predicted'], axis=-1)

        if verbose:
            print(f"Next obs real and model error L2 determined")

        doc['next_obss_real'] = next_obs_real
        doc['model_error_l2'] = model_error_l2

        json.dump(doc, open(rollout_doc_path, 'w'), default=encode_numpy, indent=3)

        if verbose:
            print(f"Updated rollout document saved to {rollout_doc_path}")
    else:
        if verbose:
            print(f"Rollout document {rollout_doc_path} already contains next_obss_real and model_error_l2")

    return doc

def add_model_error_to_rollout_docs_for_all_files_in_dir(dir_path: str, env_name: str, verbose: bool = False) -> None:
    """
        Adds model error information to all rollout documents in the specified directory.
            Args:
                dir_path: The path to the directory containing rollout documents.
                env: The name of the environment.
            Returns:
                None
    """
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            print(f"Processing file: {file_path}")
            add_model_error_to_rollout_doc(file_path, env_name, verbose=True)
            

def load_rollout_docs(rollout_doc_paths: Union[str, list], add_model_error_if_not_contained: bool = False, env:str = None, cast_to_nparray: bool = True, verbose: bool = False) -> list:
    """
        Loads rollout documents from the specified paths.
        Args:
            rollout_doc_paths: The path to the json rollout document or a list of paths.
            add_model_error_if_not_contained: Whether to add model error information if not already present.
            env: The environment name to use for model error computation.
            verbose: Whether to print verbose output.
        Returns:
            A list of loaded rollout documents.
    """
    if isinstance(rollout_doc_paths, str):
        rollout_doc_paths = [rollout_doc_paths]

    docs = []
    for path in rollout_doc_paths:
        if verbose:
            print(f"Loading rollout document {path}")
        doc = json.load(open(path, 'r'))
        if cast_to_nparray:
            for k,v in doc.items():
                if not isinstance(v, np.ndarray):
                    doc[k] = np.array(v)
        docs.append(doc)

    if add_model_error_if_not_contained:
        if env is None:
            raise ValueError("Environment name must be provided to compute model error.")
        
        for doc in docs:
            add_model_error_to_rollout_doc(doc, env, verbose=verbose)

    return docs
