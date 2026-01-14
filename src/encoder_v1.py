""" SIMPLE TRAINABLE CNN ENCODER FOR ATARI FRAMES """
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariCNNEncoder(nn.Module):
    """
    LIGHTWEIGHT CNN ENCODER FOR ATARI FRAMES (84x84 FRAMES)
    EXPECTED CORRELATION : 0.75-0.85
    """

    def __init__(
        self,
        input_channels : int = 4,
        embedding_dim : int = 384,
    ):
        super().__init__()
        """ CONVOLUTIONAL FEATURE EXTRACTOR """
        self.encoder = nn.Sequential(
            # LAYER 1 
            # 84x84 -> 20x20
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding = 0), # OUTPUT: (64, 20, 20)
            nn.ReLU(inplace=True),

            # LAYER 2 
            # 20x20 -> 9x9
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding = 0), # OUTPUT: (128, 9, 9)
            nn.ReLU(inplace=True),

            # LAYER 3 
            # 9x9 -> 7x7
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding = 0), # OUTPUT: (256, 7, 7)
            nn.ReLU(inplace=True),

            nn.Flatten(), # OUTPUT: (256*7*7) = (12544)
        )

        """ PROJECTION HEAD (SPATIAL PROCESSING)"""
        self.projection = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self._init_weights()
    

    def _init_weights(self):
        """ INITIALIZE WEIGHTS TO PREVENT VANISHING/EXPLODING GRADIENTS """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor:
        """ FORWARD PASS THROUGH ENCODER 
        INPUT : TENSOR SHAPE : (B, 4, 84, 84)
        OUTPUT : TENSOR SHAPE : (B, EMBEDDING_DIM) 
        """

        y = self.encoder(x)  # SHAPE: (B, 256*7*7)
        z = self.projection(y)  # SHAPE: (B, EMBEDDING_DIM)
        z_norm = F.normalize(z, p=2, dim=1)  # L2 NORMALIZATION
        return z_norm