import torch

def cossim_loss(x, y):
    """
    Cosine Similarity Loss
    """
    x = x.flatten(start_dim=2, end_dim=-1)
    y = y.flatten(start_dim=2, end_dim=-1)
    
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    loss = 1 - cos_sim.mean()
    
    return loss