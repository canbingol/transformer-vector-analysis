import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 256
    n_head: int = 4
    n_layers: int = 4
    block_size: int = 128
    intermediate_size: int | None = None
    head_dim: int = field(init=False)

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.head_dim = self.d_model // self.n_head

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.d_model

def get_device(device: str | torch.device = "mps") -> torch.device:
    if isinstance(device, str):
        if device.lower() == "gpu":
            device = "cuda"
        device = torch.device(device)
    return device


def apply_causal_mask(scores: torch.tensor, seq_len: int):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    mask = mask.to(scores.device)

    scores = scores.masked_fill(mask, float("-inf"))

    return scores

class FFN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.up_proj = nn.Linear(config.d_model, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model)
        self.act_fn = nn.SiLU()

    def forward(self, x:torch.Tensor):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.wo = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x:torch.Tensor):

        batch_size, seq_len, _ = x.shape # (batch_size, seq_len, dim)

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # transform
        q = q.reshape(batch_size,seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size,seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size,seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = apply_causal_mask(scores,seq_len)

        scores = F.softmax(scores.float(),dim=-1).type_as(q)   

        output = torch.matmul(scores, v)
        # (B, H_Q, seq_len, Head_Dim) -> (B, seq_len, H_Q, Head_Dim) -> (B, seq_len, Dim)
        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output)
    
class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()  

        self.attention = Attention(config=config)
        self.ffn = FFN(config=config)

        self.norm_1 = nn.RMSNorm(config.d_model)
        self.norm_2 = nn.RMSNorm(config.d_model)

    def forward(self, x:torch.Tensor):

        h = x + self.attention(self.norm_1(x))
        out = h + self.ffn(self.norm_2(h))

        return out
    
class DecoderModel(nn.Module):
    def __init__(self, config: Config, device: str | torch.device = "cpu"):
        super().__init__()  

        self.device = get_device(device)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(DecoderLayer(config=config))
        
        self.norm = nn.RMSNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.to(self.device)

    def forward(self, x:torch.Tensor):
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).float()

        return self.linear(h)

if __name__ == "__main__":
    default_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    config = Config()
    model = DecoderModel(config=config, device=default_device)

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 4),
        dtype=torch.long
    ).to(model.device)

    out = model(x)

    print(f"Device: {model.device}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    last_logits = out[:, -1, :]
    estimated_idx = torch.argmax(last_logits, dim=-1)

    print(f"Estimated idx: {estimated_idx}")
    print("OK")
