# Variational Autoencoder (VAE)

A VAE implementation from scratch in PyTorch, trained on the CelebA dataset to generate face images.

## Architecture

- **Encoder**: ResNet-18 backbone (without pretrained weights) that outputs `mu` and `log_var` vectors
- **Sampling**: Reparameterization trick to sample latent vector `z` from `N(mu, sigma)`
- **Decoder**: Transposed convolution + bilinear upsampling blocks to reconstruct 224x224 RGB images
- **Loss**: MSE reconstruction loss + KL divergence

## Project Structure

```
├── main.py                 # Entry point with argument parsing and training loop
├── model/
│   ├── vae.py              # VAE module (encoder + sampling + decoder)
│   ├── encoder.py          # ResNet-18 based encoder + reparameterization sampling
│   └── decoder.py          # Upsampling decoder with conv blocks
├── utils/
│   ├── dataloader.py       # CelebA dataset loader
│   ├── engine.py           # Training and evaluation logic
│   └── loss.py             # Reconstruction + KL divergence loss
├── data/                   # CelebA images (not included)
└── logs/                   # Generated samples saved here after each epoch
```

## Usage

Place CelebA images (`.jpg`) in `data/`, then run:

```bash
python main.py --epochs 10 --batch_size 32 --embed_dim 256 --device mps
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | `data/` | Path to image directory |
| `--batch_size` | `32` | Batch size |
| `--grad_steps` | `8` | Gradient accumulation steps |
| `--epochs` | `3` | Number of epochs |
| `--lr` | `0.0005` | Learning rate |
| `--embed_dim` | `256` | Latent space dimension |
| `--grad_clip` | `3.0` | Gradient clipping max norm |
| `--device` | `mps` | Device (`cpu`, `cuda`, `mps`) |
| `--savepath` | `logs/` | Where to save generated images |

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- Pillow
- numpy

## License

MIT
