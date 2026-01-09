""" EXTRACT 100K EMBEDDINGS FROM REPLAY BUFFER USING FROZEN DINOv2 ENCODER """

import os 
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# IMPORT FROZEN VISION ENCODER 
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from encoder import FrozenDinoV2Encoder

def load_replay_buffer(
    path : str
):
    """ LOAD REPLAY BUFFER (NPZ FILE)
    INPUT : PATH TO REPLAY BUFFER
    OUTPUT : DICTIONARY CONTAINING REPLAY BUFFER ARRAYS
    """
    
    print(f"LOADING REPLAY BUFFER FROM {path}...")
    with np.load(path, allow_pickle = True) as data:
        frames = data["frames"] # SHAPE (N, 84, 84) OR (N, 4, 84, 84)
        print(f"SHAPE: {frames.shape}")
        print(f"DATA TYPE: {frames.dtype}")
        print(f"MIN VALUE: {frames.min()}")
        print(f"MAX VALUE: {frames.max()}")
        
        assert frames.shape[0] == 100_000, f"EXPECTED 100,000 FRAMES, GOT {frames.shape[0]}"
        assert frames.shape[1:] == (4, 84, 84), f"EXPECTED SHAPE (4,84,84), GOT {frames.shape[1:]}"
    return frames

def extract_embeddings(
    frames : np.ndarray,
    encoder : FrozenDinoV2Encoder,
    batch_size : int = 64,
    max_frames : int = 100000
) -> np.ndarray:
    """ EXTRACT EMBEDDINGS FROM FRAMES USING FROZEN DINOv2 ENCODER
    INPUT : FRAMES (N, 4, 84, 84)
    OUTPUT : EMBEDDINGS (N, EMBEDDING_DIM)
    """
    
    # PREPROCESS FRAMES
    num_frames = min(len(frames), max_frames)
    print(f"EXTRACTING {num_frames} EMBEDDINGS...")
    print(f"BATCH SIZE : {batch_size}")
    
    # ALLOCATE MEMORY FOR EMBEDDINGS
    embeddings = np.zeros((num_frames, 384), dtype = np.float32) # DINOv2 BASE EMBEDDING DIM (384) = 100000 x 384 size
    num_batches = (num_frames + batch_size - 1) // batch_size # CEILING DIVISION FOR COMPLETE COVERAGE
    
    with tqdm (
        total = num_frames,
        desc = "EXTRACTING EMBEDDINGS",
        unit = "frames",
    ) as pbar:
        
        for batch_idx in range(num_batches):
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            
            # BATCH 
            batch_frames = frames[start_idx:end_idx] # SHAPE (B, 4, 84, 84)
            
            # EXTRACT EMBEDDINGS
            batch_embeddings = encoder.extract_batch(batch_frames) # SHAPE (B, 384)
            
            # STORE EMBEDDINGS
            embeddings[start_idx:end_idx] = batch_embeddings
            
            # UPDATE PROGRESS BAR
            batch_count = end_idx - start_idx
            pbar.update(batch_count)
    
    return embeddings
            
    
    
    

    
    