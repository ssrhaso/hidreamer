""" FROZEN DINOv2 ENCODER AND PREPROCESSING FOR ATARI FRAMES """

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from numpy import np
from typing import Union, Tuple

class FrozenDinoV2Encoder(nn.Module):
    """ FROZEN ViT (SMALL) ENCODER FROM DINOv2 
    
   
    INPUT : TENSOR SHAPE : (8, 3, 224, 224)
    OUTPUT : TENSOR SHAPE : (8, 384) 
    """

    def __init__(
        self,
        model_name : str = "facebook/dinov2-vit-small",
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # LOAD IMAGE PROCESSOR AND MODEL
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # FREEZING PARAMETERS
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()
        self.embedding_dim = 384 #DINOv2-VIT-SMALL EMBEDDING DIMENSION
        
        # LOGGING
        print(f"LOADED {model_name}")
        print(f"DEVICE : {device}")
        print(f"EMBEDDING DIMENSION : {self.embedding_dim}")
        print(f"FROZEN : {not any(p.requires_grad for p in self.model.parameters())}")
    
    def preprocess_frame(
        self,
        frame : np.ndarray,
    ) -> np.ndarray:
        """ PREPROCESS ATARI FRAMES FOR DINOv2 ENCODER

        INPUT : ATARI FRAME SHAPE (84, 84) , GRAYSCALE, UINT8, or 4 STACKED FRAMES SHAPE (4, 84, 84)
        OUTPUT : IMAGE ARRAY  SHAPE (84,84,3) , RGB, UINT8
        """

        # HANDLE STACKED FRAMES 
        if frame.ndim == 3 and frame.shape[0] ==  4:
            frame = frame[-1, :, :]
        
        # ASSERT SHAPE 
        assert frame.shape == (84, 84), f"EXPECTED FRAME SHAPE (84,84), GOT {frame.shape}" 

        # CONVERT UINT8 [0, 255] -> float32 [0.0, 1.0]
        frame_float = frame.astype(np.float32) / 255.0

        # CONVERT GRAYSCALE -> RGB
        frame_rgb = np.repeat(frame_float[:, :, np.newaxis], 3, axis = 2)  

        return frame_rgb
    
    def forward(
        self,
        images : Union[torch.Tensor, np.ndarray],

    ) -> torch.Tensor:
        """ EXTRACT EMBEDDINGS FROM IMAGES 
        
        INPUT : BATCH OF IMAGES
        OUTPUT: BATCH OF EMBEDDINGS
        """

        # FROZEN FORWARD PASS
        with torch.no_grad():
            # PROCESS FRAMES THROUGH DINOv2 PREPROCESSOR 
            inputs = self.processor(images = images, return_tensors = "pt").to(self.device)

            # FORWARD PASS THROUGH DINOv2 MODEL
            outputs = self.model(**inputs)

            # EXTRACT CLS TOKEN EMBEDDINGS
            embeddings = outputs.last_hidden_state[:, 0, :] # CLS TOKEN AT INDEX 0    

        return embeddings


            
def extract_batch(
    self,
    frames : np.ndarray,

) -> np.ndarray:
    """ EXTRACT DINOv2 EMBEDDINGS FROM BATCH OF FRAMES 

    INPUT : BATCH OF FRAMES SHAPE (B, 84, 84) OR (B, 4, 84, 84)
    OUTPUT : BATCH OF EMBEDDINGS SHAPE (B, 384)
    """

    # PREPROCESS FRAMES

    if frames.ndim == 3: # (B, 84, 84)
        processed = np.array([
            self.preprocess_frame(frames[i]) for i in range(len(frames))
        ])
        
    
    elif frames.ndim == 4:# (B, 4, 84, 84)
        processed = np.array([
            self.preprocess_frame(frames[i]) for i in range(len(frames))
        ])
    
    else:
        raise ValueError(f"EXPECTED FRAMES DIMENSIONS (B, 84, 84) OR (B, 4, 84, 84), GOT {frames.shape}")
    
    
        