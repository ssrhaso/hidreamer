""" TRAIN VQ TOKENIZER ON DINOv2 EMBEDDINGS """
""" NOTE: BASELINE CODE, TO BE MODIFIED """

import os 
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from vq import VQVAE, VQTokenizer, load_config



class EmbeddingDataset(Dataset):
    """ DATASET WRAPPER FOR DINOv2 EMBEDDINGS
    INPUT: .npy FILE CONTAINING PRECOMPUTED EMBEDDINGS
    """
    
    # CONSTRUCTOR
    def __init__(self, embeddings_path : str,):
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        # L2 normalize to unit length
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = self.embeddings / norms
        print(f"LOADED EMBEDDINGS FROM {embeddings_path}, LENGTH: {len(self.embeddings)}")
    
    # SAMPLING METHODS
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx])
    
def compute_codebook_stats(
    tokens : torch.Tensor,
    num_codes : int = 256,
):
    """ COMPUTE CODEBOOK USAGE STATISTICS (TOKEN FREQUENCY) """
    
    # FLATTEN TOKENS (HANDLES SPATIAL DIMENSIONS)
    tokens = tokens.flatten()
    
    # UNIQUE CODES
    unique_codes = torch.unique(tokens)
    num_used = len(unique_codes)
    
    # FREQUENCY COUNTS
    token_counts = torch.bincount(tokens, minlength = num_codes).float()
    token_probs = token_counts / token_counts.sum()
    
    # PERPLEXITY (EFFECTIVE NUMBER OF CODES USED)
    token_probs_nonzero = token_probs[token_probs > 0]
    entropy = -torch.sum(token_probs_nonzero * torch.log(token_probs_nonzero))
    perplexity = torch.exp(entropy).item()
    
    # HISTOGRAM 
    usage_histogram = token_counts.cpu().numpy().tolist()
    
    return {
        'num_used_codes': int(num_used),
        'total_codes': num_codes,
        'usage_ratio': float(num_used / num_codes),
        'perplexity': float(perplexity),
        'usage_histogram': usage_histogram,
    }


def train_vq(
    embeddings_path : str = "data/embeddings_ALE_Pong-v5_dinov2_base.npy",
    output_dir : str = "checkpoints",
    num_codes : int = 256,
    latent_dim : int = 128,
    commitment_cost : float = 0.25,
    batch_size : int = 256,
    num_epochs : int = 50,
    learning_rate : float = 1.0e-3,
    val_split : float = 0.05,
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed : int = 42,
):
    """ TRAIN VQ TOKENIZER ON DINOv2 EMBEDDINGS """
    
    # SET SEED FOR REPRODUCIBILITY
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    print(f"SAVING CHECKPOINTS TO: {output_dir}")
    print(f"TRAINING VQ TOKENIZER WITH {num_codes} CODES)")
    print(f" DEVICE    : {device}"
          f"\n BATCH SIZE: {batch_size}"
          f"\n EPOCHS    : {num_epochs}"
          f"\n LR        : {learning_rate}"
          f"\n VAL SPLIT : {val_split}")
    
    # LOAD DATASET
    dataset = EmbeddingDataset(embeddings_path)
    
    # TRAIN-VAL SPLIT
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator = torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        dataset= train_dataset, 
        batch_size=batch_size, 
        num_workers= 2, 
        pin_memory= (device == 'cuda'),)
    
    val_loader = DataLoader(
        dataset= val_dataset, 
        batch_size=batch_size, 
        num_workers= 2,
        pin_memory= (device == 'cuda'),)
    
    print(f"TRAINING SAMPLES : {len(train_dataset)} | VALIDATION SAMPLES: {len(val_dataset)}")
    
    # INIT MODEL
    model = VQTokenizer(
        input_dim = 384,
        latent_dim = latent_dim,
        num_codes = num_codes,
        commitment_cost = commitment_cost,
    ).to(device)
    
    # INITIALIZE CODEBOOK FROM DATA SAMPLES (no projection)
    print(f"INITIALIZING CODEBOOK FROM DATA...")
    with torch.no_grad():
        init_batch = next(iter(train_loader)).to(device)
        init_batch = init_batch.unsqueeze(1).unsqueeze(2)
        init_samples = init_batch.reshape(-1, 384)[:num_codes]
        
        # Repeat if needed
        if init_samples.size(0) < num_codes:
            repeats = (num_codes + init_samples.size(0) - 1) // init_samples.size(0)
            init_samples = init_samples.repeat(repeats, 1)[:num_codes]
        
        model.vq.codebook.weight.data.copy_(init_samples)
        model.vq.ema_weight.copy_(init_samples)
    print(f"CODEBOOK INITIALIZED")
    
    # NO OPTIMIZER NEEDED - EMA updates codebook without gradients
    print(f"NOTE: Using EMA updates (no gradient-based training)")
    
    """ TRAINING LOOP"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        train_tokens_all = []
        
        pbar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{num_epochs} - Training",
            leave = True
        )
        
        for batch in pbar:
            batch = batch.to(device)
            # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
            batch = batch.unsqueeze(1).unsqueeze(2)
            
            # Forward pass (EMA updates codebook automatically)
            z_quantized, loss, tokens = model(batch)
            
            train_loss += loss.item()
            train_tokens_all.append(tokens.cpu().detach())
            
            # UPDATE PROGRESS BAR
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        """ VALIDATION LOOP """
        model.eval()
        val_loss = 0.0
        val_tokens_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
                batch = batch.unsqueeze(1).unsqueeze(2)
                z_quantized, loss, tokens = model(batch)
                val_loss += loss.item()
                val_tokens_all.append(tokens.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        """ CODEBOOK ANALYSIS """
        train_tokens_all = torch.cat(train_tokens_all, dim=0)
        val_tokens_all = torch.cat(val_tokens_all, dim=0)
        
        train_stats = compute_codebook_stats(
            train_tokens_all, num_codes = num_codes
        )
        val_stats = compute_codebook_stats(
            val_tokens_all, num_codes = num_codes
        )   
        print(f"Epoch {epoch+1}/{num_epochs} Summary:"
                f"\n TRAIN LOSS: {avg_train_loss:.4f} | VAL LOSS: {avg_val_loss:.4f}"
                f"\n TRAIN CODEBOOK USAGE: {train_stats['num_used_codes']}/{num_codes} "
                f"({train_stats['usage_ratio']*100:.2f}%), PERPLEXITY: {train_stats['perplexity']:.2f}"
                f"\n VAL CODEBOOK USAGE  : {val_stats['num_used_codes']}/{num_codes} "
                f"({val_stats['usage_ratio']*100:.2f}%), PERPLEXITY: {val_stats['perplexity']:.2f}"
            )
        
        """ SAVE CHECKPOINT IF BEST """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'vq_model_best.pth'))
            print(f" Saved Best Model Checkpoint")
        
    # FINAL TOKENIZATION
    print(f"GENERATING FINAL TOKENS FOR FULL DATASET...")
    
    model.eval()
    all_tokens = []
    full_loader = DataLoader(
        dataset= dataset, 
        batch_size=batch_size, 
        num_workers= 2,
        pin_memory= (device == 'cuda'),)
    
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Tokenizing Full Dataset"):
            batch = batch.to(device)
            # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
            batch = batch.unsqueeze(1).unsqueeze(2)
            tokens = model.encode(batch)
            all_tokens.append(tokens.cpu())
    
    all_tokens = torch.cat(all_tokens, dim=0).numpy()
    
    """ FINAL STATISTICS """
    final_stats = compute_codebook_stats(
        torch.from_numpy(all_tokens), num_codes = num_codes
    )
    
    with torch.no_grad():
        sample_batch = next(iter(full_loader)).to(device)
        # Add spatial dimensions: [B, 384] -> [B, 1, 1, 384]
        sample_batch = sample_batch.unsqueeze(1).unsqueeze(2)
        z_quantized, _, _ = model(sample_batch)
        quantized_stats = {
            'mean': float(z_quantized.mean().item()),
            'std': float(z_quantized.std().item()),
            'min': float(z_quantized.min().item()),
            'max': float(z_quantized.max().item()),
        }
        
    # HEALTH CHECK
    print(f"FINAL CODEBOOK USAGE: {final_stats['num_used_codes']}/{num_codes} "
          f"({final_stats['usage_ratio']*100:.2f}%), PERPLEXITY: {final_stats['perplexity']:.2f}"
    )
    print(f"QUANTIZED LATENT STATS: MEAN={quantized_stats['mean']:.4f}, "
          f"STD={quantized_stats['std']:.4f}, MIN={quantized_stats['min']:.4f}, MAX={quantized_stats['max']:.4f}"
    )
    
    # COLLAPSE WARNING
    if final_stats['num_used_codes'] < num_codes * 0.1:
        print("WARNING: CODEBOOK COLLAPSE DETECTED! FEW CODES USED.")
    else:
        print("CODEBOOK USAGE HEALTHY.")
    
    """ SAVE FINAL TOKENS AND STATS """
    print(f"SAVING FINAL TOKENS AND TRAINING STATS...")
    tokens_path = os.path.join(output_dir, 'vq_tokens_100k.npy')
    model_path = os.path.join(output_dir, 'vq_model_final.pth')
    stats_path = os.path.join(output_dir, 'vq_stats.json')
    
    np.save(tokens_path, all_tokens)
    torch.save(model.state_dict(), model_path)
    
    stats = {
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
        'final_codebook_stats': final_stats,
        'quantized_latent_stats': quantized_stats,
        'hyperparameters': {
            'num_codes': num_codes,
            'latent_dim': latent_dim,
            'commitment_cost': commitment_cost,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'val_split': val_split,
            'seed': seed,
        }
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"TRAINING COMPLETE. SAVED TOKENS TO {tokens_path}, MODEL TO {model_path}, STATS TO {stats_path}")
    
    return model, all_tokens, stats

if __name__ == "__main__":
    
    config = load_config("configs/vq.yaml")
    
    train_vq(
        embeddings_path = config['data']['embeddings_path'],
        output_dir = config['training'].get('save_dir', 'checkpoints'),
        num_codes = config['model']['num_codes'],
        latent_dim = config['model']['latent_dim'],
        commitment_cost = config['model']['commitment_cost'],
        batch_size = config['training']['batch_size'],
        num_epochs = config['training']['num_epochs'],
        learning_rate = config['training']['learning_rate'],
        val_split = config['training']['val_split'],
        seed = config['seed'],
    )

