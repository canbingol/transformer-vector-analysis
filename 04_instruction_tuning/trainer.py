import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from model import Config, DecoderModel

from prepare_dataset import prepare_it_data
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

config = Config()
default_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
pretrained_ckpt_path = os.path.join(checkpoints_dir, "pretrained_decoder.pt")

pretrained_checkpoint = torch.load(pretrained_ckpt_path, map_location=default_device, weights_only=False)
pretrained_config = pretrained_checkpoint["config"]
if isinstance(pretrained_config, dict):
    pretrained_config = Config(**pretrained_config)
model = DecoderModel(pretrained_config, device=default_device)
model.load_state_dict(pretrained_checkpoint["model_state_dict"])
print("Pretrained Model Loaded")

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
                y.view(-1),
                ignore_index=-100

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

def trainer(model: DecoderModel = model, num_epoch: int = 2, lr: float = 3e-4, min_lr: float = 3e-5, device: str | torch.device | None = None):
    if device is None:
        device = default_device
    count_parameters(model)

    print(f"Device: {device}")
    # Create losses directory if it doesn't exist
    os.makedirs("losses", exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize CSV files with headers
    with open("losses/train_losses.txt", "w") as f:
        f.write("global_step,loss\n")
    with open("losses/val_losses.txt", "w") as f:
        f.write("global_step,loss\n")

    train_loader, val_loader = prepare_it_data(batch_size=8)

    print(f"Len Train Loader: {len(train_loader)}")
    print(f"Len Val Loader: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1
    )

    train_losses, val_losses = [], []
    global_step = 0

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
            global_step += 1
            is_last_step = (batch_idx == len(train_loader) - 1)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update tqdm with current train loss
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            # Save train loss to file
            with open("losses/train_losses.txt", "a") as f:
                f.write(f"{global_step},{loss.item()}\n")

            if global_step % 500 == 0:
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append(val_loss)
                print(f"Epoch {epoch + 1} Global Step: {global_step} Val Loss {val_loss}")
                # Save val loss to file
                with open("losses/val_losses.txt", "a") as f:
                    f.write(f"{global_step},{val_loss}\n")
            
    model.eval()
    final_val_loss = estimate_loss(model, val_loader, device, max_batches=len(val_loader))
    print(f"Full Val Loss {final_val_loss}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
            "step": global_step,
            "final_train_loss": train_losses[-1] if len(train_losses) > 0 else None,
            "final_val_loss": final_val_loss,
        },
        os.path.join(checkpoints_dir, "instruction_tuned_decoder.pt")
    )
    print(f"Trained model saved as {os.path.join(checkpoints_dir, 'instruction_tuned_decoder.pt')}")

if __name__ == "__main__":
    trainer()