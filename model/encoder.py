import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

class Encoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        model = get_model("resnet18", weights=None)
        self.model = nn.Sequential(*model.children())[:-1]

        self.mu_head = nn.Linear(in_features=512, out_features=embed_dim)
        self.log_var_head = nn.Linear(in_features=512, out_features=embed_dim) 
        

    def forward(self, x: torch.Tensor):
        x = self.model(x).reshape(x.shape[0], -1)
        mu = self.mu_head(x)
        log_var = self.log_var_head(x)

        return mu, log_var


class Sampling(nn.Module):
    def __init__(self, eps_mu: float = 0.0, eps_sigma: float = 1.0) -> None:
        super().__init__()
        self.eps_mu = eps_mu
        self.eps_sigma = eps_sigma

    def forward(self, mu, log_var):
        bs, embed_dim = mu.shape
        
        sigma = torch.exp(0.5 * log_var)
        eps = torch.normal(mean=self.eps_mu, std=self.eps_sigma, size=(bs, embed_dim))
        z = mu + sigma * eps
        
        return z.reshape(bs, embed_dim, 1, 1)

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    encoder = Encoder(embed_dim=256)
    sample = Sampling()
    mu, log_var = encoder(x)
    z = sample(mu, log_var)
    
    print(f"x: {x.shape}, (mu, log_var): ({mu.shape}, {log_var.shape}, z: {z.shape}")

    print(encoder)
    