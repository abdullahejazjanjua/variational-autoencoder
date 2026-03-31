import math

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.initial_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=7),
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