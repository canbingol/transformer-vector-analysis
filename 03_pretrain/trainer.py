import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from model import Config, DecoderModel

from prepare_dataset import prepare_pretrain_data
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

config = Config()
default_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = DecoderModel(config=config, device=default_device)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters (M): {total_params / 1e6:.2f}M")
    print(f"Trainable parameters (M): {trainable_params / 1e6:.2f}M")

    return total_params, trainable_params

def estimate_loss(model, val_loader, device, max_batches=500):
    model.eval()
    losses = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def calculate_loss(model,input_batch,target_batch,device):

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # Forward pass
    logits = model(input_batch)
    del input_batch

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_batch.reshape(-1),
        ignore_index= -100
    )

    del target_batch, logits
    return loss

def trainer(model: DecoderModel = model, num_epoch: int = 1, lr: float = 3e-4, min_lr: float = 3e-5, device: str | torch.device | None = None):
    if device is None:
        device = default_device
    count_parameters(model)

    print(f"Device: {device}")
    # Create losses directory if it doesn't exist
    os.makedirs("losses", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize CSV files with headers
    with open("losses/train_losses.txt", "w") as f:
        f.write("step,Loss\n")
    with open("losses/val_losses.txt", "w") as f:
        f.write("step,Loss\n")

    # Calculate initial validation loss and save initial model
    train_loader, val_loader = prepare_pretrain_data(batch_size=180)
    initial_val_loss = estimate_loss(model, val_loader, device, max_batches=len(val_loader))
    print(f"Initial model Full Val Loss: {initial_val_loss}")


    # Save initial model before training
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "epoch": 0,
            "step": -1,
            "final_train_loss": None,
            "final_val_loss": initial_val_loss,
        },
        "checkpoints/random_decoder.pt"
    )
    print("Initial model saved as checkpoints/random_decoder.pt")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1
    )

    train_losses, val_losses = [], []
    step = -1

    for epoch in range(num_epoch):
        model.train()


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader),
            eta_min=min_lr
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}")
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            loss = calculate_loss(model, input_batch, target_batch, device)
            loss.backward()
            train_losses.append(loss.item())
            step += 1
            is_last_step = (batch_idx == len(train_loader) - 1)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update tqdm with current train loss
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            # Save train loss to file
            with open("losses/train_losses.txt", "a") as f:
                f.write(f"{step},{loss.item()}\n")

            if step % 1000 == 0:
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append(val_loss)
                print(f"Epoch {epoch + 1} Step: {step} Val Loss {val_loss}")
                # Save val loss to file
                with open("losses/val_losses.txt", "a") as f:
                    f.write(f"{step},{val_loss}\n")
            
    model.eval()
    val_loss = estimate_loss(model, val_loader, device, max_batches=len(val_loader))
    print(f"Epoch {epoch + 1} Step: {step} Full Val Loss {val_loss}")
    final_val_loss = val_loss

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
            "step": step,
            "final_train_loss": train_losses[-1] if len(train_losses) > 0 else None,
            "final_val_loss": final_val_loss,
        },
        "checkpoints/pretrained_decoder.pt"
    )
    print("Trained model saved as checkpoints/pretrained_decoder.pt")

if __name__ == "__main__":
    trainer()