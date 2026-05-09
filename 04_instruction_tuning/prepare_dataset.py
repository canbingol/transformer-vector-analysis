import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


DATASET_ID = "tatsu-lab/alpaca"


def prepare_it_data(
    batch_size,
    max_seq_len=256,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
    val_ratio=0.05,
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.model_max_length = int(1e9)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.train_test_split(test_size=val_ratio, seed=42)

    train_data = dataset["train"]
    val_data = dataset["test"]

    class InstructionDataset(Dataset):
        def __init__(self, hf_dataset, max_seq_len):
            super().__init__()

            self.input_ids = []
            self.target_ids = []

            for text in hf_dataset["text"]:
                prompt, answer = self.convert_text(text)

                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False
                )["input_ids"]

                answer_ids = tokenizer(
                    answer,
                    add_special_tokens=False
                )["input_ids"]

                full_ids = prompt_ids + answer_ids

                input_ids = full_ids[:-1]
                target_ids = [-100] * (len(prompt_ids) - 1) + answer_ids

                input_ids = input_ids[:max_seq_len]
                target_ids = target_ids[:max_seq_len]

                if len(input_ids) < max_seq_len:
                    pad_len = max_seq_len - len(input_ids)

                    input_ids += [pad_token_id] * pad_len
                    target_ids += [-100] * pad_len

                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_ids, dtype=torch.long))

        def convert_text(self, text):
            instruction_block = text.split("### Instruction:")[1].split("### Response:")[0].strip()
            response_part = text.split("### Response:")[1].strip()

            if "### Input:" in instruction_block:
                instruction_part = instruction_block.split("### Input:")[0].strip()
                input_part = instruction_block.split("### Input:")[1].strip()

                prompt = (
                    "System:\n"
                    "You are a helpful assistant.\n\n"
                    "Instruction:\n"
                    f"{instruction_part}\n\n"
                    "Input:\n"
                    f"{input_part}\n\n"
                    "Response:\n"
                )
            else:
                prompt = (
                    "System:\n"
                    "You are a helpful assistant.\n\n"
                    "Instruction:\n"
                    f"{instruction_block}\n\n"
                    "Response:\n"
                )

            answer = response_part + tokenizer.eos_token

            return prompt, answer

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            x = self.input_ids[idx]
            y = self.target_ids[idx]

            return x, y

    train_dataset = InstructionDataset(
        hf_dataset=train_data,
        max_seq_len=max_seq_len
    )

    val_dataset = InstructionDataset(
        hf_dataset=val_data,
        max_seq_len=max_seq_len
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    train_loader, val_loader, tokenizer = prepare_it_data(
        batch_size=4,
        max_seq_len=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False
    )

    x, y = next(iter(train_loader))

    print("Input shape :", x.shape)
    print("Target shape:", y.shape)

    print("\nInput ids:")
    print(x[0])

    print("\nTarget ids:")
    print(y[0])

    decoded_input = tokenizer.decode(x[0], skip_special_tokens=False)

    target_tokens = y[0][y[0] != -100]
    decoded_target = tokenizer.decode(target_tokens, skip_special_tokens=False)

    print("\nDecoded input:")
    print(decoded_input)

    print("\nDecoded target / loss computed part:")
    print(decoded_target)