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
        frames = data["states"] # SHAPE (N, 84, 84) OR (N, 4, 84, 84)
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
    
    # PRE LOOP SETUP
    embeddings = np.zeros((num_frames, 384), dtype = np.float32) # DINOv2 BASE EMBEDDING DIM (384) = 100000 x 384 size
    num_batches = (num_frames + batch_size - 1) // batch_size # CEILING DIVISION FOR COMPLETE COVERAGE
    
    """ EXTRACTION LOOP """
    with tqdm (
        total = num_frames,
        desc = "EXTRACTING EMBEDDINGS",
        unit = "frames",
    ) as pbar:
        
        for batch_idx in range(num_batches):
            
            # BATCH INDICIES (0 - 63, 64 - 127, ...)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            
            # BATCH 
            batch_frames = frames[start_idx:end_idx] # SHAPE (B, 4, 84, 84)
            
            # EXTRACT EMBEDDINGS
            batch_embeddings = encoder.extract_batch(batch_frames) # SHAPE (B, 384)
            
            # STORE EMBEDDINGS (MOVE FROM GPU TO CPU TO NUMPY)
            embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
            
            # UPDATE PROGRESS BAR
            batch_count = end_idx - start_idx
            pbar.update(batch_count)
    
    return embeddings
            
def validate_embeddings(
    embeddings : np.ndarray,
    save_path : str
): 
    """ VALIDATE AND PREPARE SAVING EMBEDDINGS TO DISK """
    
    print(f"VALIDATING EMBEDDINGS...")
    print(f"SHAPE: {embeddings.shape}")
    print(f"DATA TYPE: {embeddings.dtype}")
    print(f"MIN VALUE: {embeddings.min()}")
    print(f"MAX VALUE: {embeddings.max()}")
    print(f"MEAN VALUE: {embeddings.mean()}")
    
    # ZERO VECTORS
    num_zero_rows = np.sum(np.all(embeddings == 0, axis = 1))
    print(f"NUMBER OF ZERO VECTORS: {num_zero_rows}")
    if num_zero_rows > 0:
        print(f"WARNING: {num_zero_rows} ZERO VECTORS DETECTED")
    
    # VARIANCE
    row_variance = np.var(embeddings, axis = 1).mean()
    print(f"AVERAGE ROW VARIANCE: {row_variance}")
    if row_variance < 1e-6:
        print(f"WARNING: LOW VARIANCE DETECTED")
    
    # FILE SIZE
    file_size_mb = ((embeddings.nbytes) / (1024 * 1024))
    print(f"FILE SIZE: {file_size_mb:.2f} MB")  
    print(f"VALIDATIONS COMPLETE. READY FOR SAVING EMBEDDINGS TO {save_path}...")
    

def main():
    """ MAIN EXECUTION """
    print("EMBEDDING EXTRACTION PROCESS STARTED...")
    
    # PATHS
    replay_buffer_path = "data/replay_buffer_ALE_Pong-v5.npz"
    output_path = "data/embeddings_ALE_Pong-v5_dinov2_base.npy"
    
    # VALIDATION
    if not os.path.exists(replay_buffer_path):
        raise FileNotFoundError(f"REPLAY BUFFER FILE NOT FOUND AT {replay_buffer_path}")
    
    # LOAD REPLAY BUFFER
    frames = load_replay_buffer(replay_buffer_path)
    
    # ENCODER INITIALIZATION
    encoder = FrozenDinoV2Encoder()
    # EMBEDDINGS
    embeddings = extract_embeddings(
        frames = frames,
        encoder = encoder,
        batch_size = 64,
        max_frames = 100_000
    )
    
    # VALIDATE EMBEDDINGS
    validate_embeddings(
        embeddings = embeddings,
        save_path = output_path
    )
    
    # SAVE EMBEDDINGS TO DISK
    print(f"SAVING EMBEDDINGS TO {output_path}...")
    np.save(output_path, embeddings)
    print("EMBEDDINGS SAVED SUCCESSFULLY.")

if __name__ == "__main__":
    main()



