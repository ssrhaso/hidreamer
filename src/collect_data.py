import gymnasium as gym
import numpy as np
import os
from typing import Dict, List, Tuple
import time

class ReplayBuffer:
    def __init__(
        self,
        max_size : int = 100_000, # 100K 
        state_shape : Tuple = (84, 84, 3), # RGB IMAGE SIZE

    ):
        self.max_size = max_size,
        self.state_shape = state_shape,
        self.ptr = 0, 
        self.size = 0,
        self.full = False,

        # PRE ALLOCATING MEMORY NUMPY ARRAYS FOR EFFICIENCY 
        self.states = np.zeros( (max_size, *state_shape), dtype = np.uint8) #(shape, dtype)
        self.actions = np.zeros(max_size, dtype = np.int32)
        self.rewards = np.zeros(max_size, dtype = np.float32)
        self.next_states = np.zeros( (max_size, *state_shape), dtype = np.uint8)
        self.dones = np.zeros(max_size, dtype = bool)

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
    ):
        
        """ STORE SINGLE TRANSITION IN THE BUFFER (state, action, reward, next_state, done), UPDATE POINTER AND FULL FLAG """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done


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
        )
        print(f"REPLAY BUFFER SAVED TO {filepath}")


    def __len__(self):

        """ RETURN SIZE OF BUFFER, OR MAX SIZE IF FULL """
        return self.max_size if self.full else self.ptr
    


    


    