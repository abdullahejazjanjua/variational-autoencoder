import torch.nn as nn

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    args
    ):

    total_len_dataset = len(dataloader)
    total_loss = 0
    for img_idx, imgs in enumerate(dataloader):

        sub_batch_size = args.batch_size // args.grad_steps

        optimizer.zero_grad()
        for i in range(args.grad_steps):
            start_idx = i * sub_batch_size
            end_idx = start_idx + sub_batch_size

            img = imgs[start_idx:end_idx, ...].to(args.device)
            out, mu, log_var = model(img)
            loss = criterion(img, out, log_var, mu)

            loss = loss / args.grad_steps
            total_loss += loss.item()
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if img_idx % args.print_freq == 0 or img_idx == total_len_dataset - 1:
            print(f"   [{img_idx}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)}")

    return total_loss
            

