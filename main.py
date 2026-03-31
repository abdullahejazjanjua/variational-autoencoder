import os
import time
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.dataloader import CelebA
from utils.loss import Criterion
from model.vae import VAE

from utils.engine import train_one_epoch, evaluate_after_one_epoch

def args_parser():
    parser = argparse.ArgumentParser(description="VAE parametres")
    
    parser.add_argument("--dataset_path", default="data/", type=str)
    
    # Training specifications
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--grad_steps", default=8, type=int, help="For simulating higher batch sizes")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=0.0005, type=float, help="Base learning rate")
    parser.add_argument("--embed_dim", default=256, type=int, help="latent dimension of z")
    parser.add_argument("--grad_clip", default=3.0, type=float)

    # Additional parametres
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--savepath", default="logs/", type=str)


    return parser


def main(args):

    model = VAE(args.embed_dim)
    optimizer = AdamW(params=model.parameters())
    
    dataset = CelebA(imgs_path=args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=(args.batch_size * args.grad_steps), shuffle=True, num_workers=args.num_workers)
    
    print(f"Using device: {args.device}")
    print(f"Total Images: {len(dataset)}")
    print("\nUsing Arguments")
    print(args)
    
    model = model.to(args.device)
    criterion = Criterion().to(args.device)        

    print("\nStarting training")
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch}]: ")
        start = time.time()
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            args=args,
        )
        end = time.time()
        print("Average stats:")
        print(f"    loss: {loss}, time: {(end-start):.4f}s")
        print("Starting Evaluating:")
        evaluate_after_one_epoch(model, savepath=args.savepath, device=args.device, num_samples=2, embed_dim=args.embed_dim, current_epoch=epoch)

if __name__ == "__main__":
    args = args_parser().parse_args()
    os.makedirs(args.savepath, exist_ok=True)
    main(args)