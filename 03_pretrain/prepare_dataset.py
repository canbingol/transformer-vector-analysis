import os
import torch, gc

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

DATASET_ID = "roneneldan/TinyStories"

def create_token_file(tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    dataset = load_dataset(DATASET_ID)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"].select(range(2500))

    eos_token_id = tokenizer.eos_token_id
    batch_size = 512

    for split_dataset, name in zip(
        [train_dataset, val_dataset],
        ["train", "validation"]
    ):
        file_path = f"data/{name}.bin"
        with open(file_path, "wb") as out_file:
            for start_idx in tqdm(
                range(0, len(split_dataset), batch_size),
                desc=f"Processing {name} dataset"
            ):
                batch_texts = split_dataset["text"][start_idx:start_idx + batch_size]
                batch_tokens = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )["input_ids"]

                for tokens in batch_tokens:
                    if eos_token_id is not None:
                        tokens.append(eos_token_id)
                    np.array(tokens, dtype=np.uint16).tofile(out_file)

        token_count = os.path.getsize(file_path) // 2
        print(f"{name}.bin saved")
        print(f"Number of tokens: {token_count}", end="\n--------------------\n")


def prepare_pretrain_data(batch_size, max_seq_len=128, pad_token=0, shuffle=False, drop_last=True, num_workers=0, pin_memory=True):

    class PretrainDataset(Dataset):
        def __init__(self, mode, max_seq_len, pad_token):
            super().__init__()


            self.input_ids, self.target_ids = [], []

            file = "data/train.bin" if mode == "train" else "data/validation.bin"
            tokens = np.memmap(file, dtype=np.uint16, mode="r")

            for i in range(0,len(tokens) -1 ,max_seq_len):
                input_ = tokens[i: i + max_seq_len]
                target_ = tokens[i + 1: i + max_seq_len]
                target_ = np.concatenate((target_,np.array([pad_token],dtype= np.uint16)))

                if len(input_) < max_seq_len:
                    continue

                self.input_ids.append(torch.tensor(input_))
                self.target_ids.append(torch.tensor(target_))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            x = torch.as_tensor(self.input_ids[idx], dtype=torch.long)
            y = torch.as_tensor(self.target_ids[idx], dtype=torch.long)
            
            return x, y


    train_dataset = PretrainDataset(mode= "train", max_seq_len= max_seq_len, pad_token= pad_token)

    val_dataset = PretrainDataset(mode= "validation", max_seq_len= max_seq_len, pad_token= pad_token)

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers,
        pin_memory = pin_memory
    )


    return train_dataloader, val_dataloader

if __name__ == "__main__":
    create_token_file(tokenizer_id="gpt2")
    td, vd = prepare_pretrain_data(batch_size=1)
    print("Data Loader")
    print(f"len train loader {len(td)}")
    print(f"len val loader {len(vd)}")
