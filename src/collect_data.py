""" DATA COLLECTION FOR ATARI GAME ENVIRONMENT (EFFICIENT STORAGE IN REPLAY BUFFER) """
import ale_py
import gymnasium as gym
import numpy as np
import os
from typing import Dict, List, Tuple
import time
import yaml


class ReplayBuffer:
    def __init__(
        self,
        max_size : int = 100_000, # 100K 
        state_shape : Tuple = (4, 84, 84), # STACKED FRAMES SHAPE

    ):
        self.max_size = max_size
        self.state_shape = state_shape
        self.ptr = 0 
        self.size = 0
        self.full = False

        # PRE ALLOCATING MEMORY NUMPY ARRAYS FOR EFFICIENCY 
        self.states = np.zeros( (max_size, *state_shape), dtype = np.uint8) #(shape, dtype)
        self.actions = np.zeros(max_size, dtype = np.int32)
        self.rewards = np.zeros(max_size, dtype = np.float32)
        self.next_states = np.zeros( (max_size, *state_shape), dtype = np.uint8)
        self.dones = np.zeros(max_size, dtype = bool)

        # EPISODE BOUNDARY TRACKING (for temporal differencing)
        self.episode_starts = []  # List of indices where episodes start

        # EPISODES
        self.episode_returns = []
        self.episode_lengths = []


    def push(
        
        self,
        state,
        action,
        reward,
        next_state,
        done,
        is_episode_start : bool = False,
    ):
        """ STORE SINGLE TRANSITION IN THE BUFFER (state, action, reward, next_state, done), UPDATE POINTER AND FULL FLAG 
        
        Args:
            is_episode_start: True if this is the first frame of a new episode
        """

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # TRACK EPISODE BOUNDARIES
        if is_episode_start:
            self.episode_starts.append(self.ptr)


        self.ptr = (self.ptr + 1) % self.max_size # CIRCULAR BUFFER
        if self.ptr == 0:
            self.full = True


    def save(
        self,
        filepath : str,
    ):
        """ SAVE THE REPLAY BUFFER DATA TO FILEPATH AS NPZ COMPRESSED FILE """

        os.makedirs(os.path.dirname(filepath), exist_ok = True)
        np.savez_compressed(
            filepath,
            states = self.states,
            actions = self.actions,
            rewards = self.rewards,
            next_states = self.next_states,
            dones = self.dones,
            episode_starts = np.array(self.episode_starts),  # Save episode boundaries
        )
        print(f"REPLAY BUFFER SAVED TO {filepath}")
        print(f"SAVED {len(self.episode_starts)} EPISODE BOUNDARIES")


    def __len__(self):
        """ RETURN SIZE OF BUFFER, OR MAX SIZE IF FULL """
        return self.max_size if self.full else self.ptr
     


def collect_episodes(
    env_name : str,
    num_episodes : int,
    target_transitions,
    save_path : str,
):
    """ COLLECT EPISODES FROM THE ENVIRONMENT AND STORE THEM IN A REPLAY BUFFER """
    
    print(f"COLLECTING {num_episodes} EPISODES FROM {env_name}...")
    env = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip = 1, screen_size = 84, grayscale_obs = True, scale_obs = False)
    env = gym.wrappers.FrameStackObservation(env, stack_size = 4)

    obs_shape = env.observation_space.shape
    print(f"OBSERVATION SHAPE: {obs_shape}")

    buffer = ReplayBuffer(max_size = target_transitions, state_shape = obs_shape)

    # INITIAL COUNTERS
    total_transitions = 0
    start_time = time.time()

    # EPISODE COLLECTION LOOP
    for episode_num in range(num_episodes):
        
        obs, _ = env.reset()
        episode_return = 0
        is_first_frame = True  # Track first frame of episode

        while True:
            # ACTION (RANDOM)
            action = env.action_space.sample() 

            # STEP
            next_obs, reward, terminated, truncated, _ = env.step(action) 
            
            # DONE FLAG
            done = terminated or truncated

            # STORE TRANSITION IN REPLAY BUFFER (ENSURE PROPER DATA TYPES FOR STORAGE EFFICIENCY)
            buffer.push(
                state = obs.astype(np.uint8),
                action = int(action),
                reward = float(reward),
                next_state = next_obs.astype(np.uint8),
                done = bool(done),
                is_episode_start = is_first_frame,  # Mark episode boundaries
            )
            
            is_first_frame = False  # Only first frame is episode start

            # UPDATE COUNTERS
            total_transitions += 1
            obs = next_obs
            episode_return += reward

            if done or total_transitions >= target_transitions:
                break
        
        # LOGGING FOR EACH EPISODE
        print(f"EPISODE {episode_num + 1}/{num_episodes} | RETURN: {episode_return:.2f} | TOTAL TRANSITIONS: {total_transitions}")

        if total_transitions >= target_transitions:
            break

    # SAVE REPLAY BUFFER TO DISK
    buffer.save(save_path)
    return buffer, buffer.episode_returns


def load_config(
    config_path : str,
):
    """ LOAD YAML CONFIG FILE FROM PATH """
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config


if __name__ == "__main__":
    config_path = "configs/data.yaml"

    cfg = load_config(config_path)
    data_cfg = cfg['data']
    safe_env_name = cfg['data']['env_name'].replace("/", "_")
    save_path = os.path.join(cfg['logging']['save_dir'], f"replay_buffer_{safe_env_name}.npz")

    collect_episodes(
        env_name = data_cfg['env_name'],
        num_episodes = data_cfg['num_episodes'],
        target_transitions = data_cfg['total_transitions'],
        save_path = save_path
    )
