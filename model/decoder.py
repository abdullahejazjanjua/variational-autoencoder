import math

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    In my notes, the decoder is defined as an explicit gaussian that uses two NNs to get mean and variance
    While implementating, we have a single network that directly gives back the image.

    1. The Probabilistic Requirement
    To train a VAE, you must optimize the Evidence Lower Bound (ELBO). The reconstruction term in the ELBO 
    requires computing the log-likelihood of the data given the latent variables: log p(x|z). To compute a 
    log-likelihood, p(x|z) must be a formal probability distribution. For continuous image data, a Gaussian 
    distribution is the standard choice: p(x|z) = N(mu(z), sigma(z)).

    2. The Implementation Shortcut (Fixing the Variance)
    In practice, it is common to simplify this model by assuming the variance is a fixed constant across all 
    pixels, typically the identity matrix I. Because the variance is fixed, you do not need a second neural 
    network to compute it. The single decoder neural network only needs to compute the mean, mu(z). Therefore, 
    when your code "gives back the image," the neural network is actually outputting the mean of the 
    Gaussian.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.initial_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=4),
                nn.BatchNorm2d(num_features=embed_dim),
                nn.ReLU(inplace=True)
            )
        blocks = []
        num_blocks = int(math.log2(embed_dim // 8))
        for i in range(num_blocks):
            in_dim = embed_dim // (2 ** i)
            out_dim = embed_dim // (2 ** (i + 1))
            blocks.append(
                DecoderBlock(in_channels=in_dim, out_channels=out_dim)
            )
        self.blocks = nn.Sequential(*blocks)
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=7, padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        num_blocks = len(self.blocks)

        x = self.initial_block(x)
        for i in range(num_blocks):
            x = self.blocks[i](x)
        x = self.final_block(x)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding="same")
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(self.upsample(x))))
    
if __name__ == "__main__":
    z = torch.randn(size=(1, 256, 1, 1))
    decoder = Decoder(embed_dim=256)
    out = decoder(z)
    print(f"out: {out.shape}")