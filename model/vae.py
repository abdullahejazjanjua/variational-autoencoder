import torch
import torch.nn as nn

from .encoder import Encoder, Sampling
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        out = self.decoder(z)

        return out, mu, log_var
    
    @torch.no_grad
    def generate(self, z: torch.Tensor):
        return self.decoder(z)

    @torch.no_grad
    def inference(self, z: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return z



if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = VAE(embed_dim=256)
    out = model(x)

    print(f"x: {x.shape} & reconstructed x: {out.shape}")