import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "teknium/OpenHermes-2.5"


def prepare_it_data(
    batch_size,
    max_seq_len=256,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
    val_ratio=0.05,
    cache_dir="cache",
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.model_max_length = int(1e9)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id

    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, f"openhermes_train_seq{max_seq_len}.pt")
    val_cache_path = os.path.join(cache_dir, f"openhermes_val_seq{max_seq_len}.pt")

    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.train_test_split(test_size=val_ratio, seed=42)

    train_data = dataset["train"]
    val_data = dataset["test"]

    class InstructionDataset(Dataset):
        def __init__(self, hf_dataset, max_seq_len, cache_path):
            super().__init__()

            if os.path.exists(cache_path):
                print(f"Loading cached dataset from: {cache_path}")
                cached_data = torch.load(cache_path)
                self.input_ids = cached_data["input_ids"]
                self.target_ids = cached_data["target_ids"]
                return

            print(f"Tokenizing dataset and saving to: {cache_path}")

            self.input_ids = []
            self.target_ids = []

            skipped_long_text = 0
            skipped_long_tokens = 0
            skipped_invalid = 0

            prompts = []
            answers = []

            for conversations in tqdm(hf_dataset["conversations"], desc="Building prompts"):
                result = self.convert_conversation(conversations)

                if result is None:
                    skipped_invalid += 1
                    continue

                prompt, answer = result

                # Kaba filtre: çok uzun textleri tokenize etmeden önce at.
                # İngilizce GPT-2 BPE için 1 token yaklaşık 3-5 karakter olabilir.
                # max_seq_len * 8 güvenli-ish bir kaba sınırdır.
                if len(prompt) + len(answer) > max_seq_len * 8:
                    skipped_long_text += 1
                    continue

                prompts.append(prompt)
                answers.append(answer)

            prompt_tokenized = tokenizer(
                prompts,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            answer_tokenized = tokenizer(
                answers,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            for prompt_ids, answer_ids in tqdm(
                zip(prompt_tokenized, answer_tokenized),
                total=len(prompt_tokenized),
                desc="Creating tensors"
            ):
                full_ids = prompt_ids + answer_ids

                # Kesin filtre: sequence max_seq_len'e sığmıyorsa at.
                # Truncate yok.
                if len(full_ids) - 1 > max_seq_len:
                    skipped_long_tokens += 1
                    continue

                input_ids = full_ids[:-1]
                target_ids = [-100] * (len(prompt_ids) - 1) + answer_ids

                if len(input_ids) < max_seq_len:
                    pad_len = max_seq_len - len(input_ids)

                    input_ids += [pad_token_id] * pad_len
                    target_ids += [-100] * pad_len

                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_ids, dtype=torch.long))

            torch.save(
                {
                    "input_ids": self.input_ids,
                    "target_ids": self.target_ids,
                },
                cache_path
            )

            print(f"Loaded examples: {len(self.input_ids)}")
            print(f"Skipped invalid examples: {skipped_invalid}")
            print(f"Skipped by rough text length: {skipped_long_text}")
            print(f"Skipped by token length: {skipped_long_tokens}")

        def convert_conversation(self, conversations):
            human_message = None
            assistant_message = None

            for message in conversations:
                role = message.get("from")
                value = message.get("value")

                if value is None:
                    continue

                if role == "human" and human_message is None:
                    human_message = value

                elif role == "gpt" and human_message is not None:
                    assistant_message = value
                    break

            if human_message is None or assistant_message is None:
                return None

            prompt = (
                "System:\n"
                "You are a helpful assistant.\n\n"
                "User:\n"
                f"{human_message.strip()}\n\n"
                "Assistant:\n"
            )

            answer = assistant_message.strip() + tokenizer.eos_token

            return prompt, answer

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            x = self.input_ids[idx]
            y = self.target_ids[idx]
            return x, y

    train_dataset = InstructionDataset(
        hf_dataset=train_data,
        max_seq_len=max_seq_len,
        cache_path=train_cache_path
    )

    val_dataset = InstructionDataset(
        hf_dataset=val_data,
        max_seq_len=max_seq_len,
        cache_path=val_cache_path
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

    return train_dataloader, val_dataloader, tokenizer


if __name__ == "__main__":

    train_loader, val_loader, tokenizer = prepare_it_data(
        batch_size=4,
        max_seq_len=256,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        cache_dir="cache"
    )
    print(f"len trian loader: {len(train_loader)}")
    print(f"len val loader: {len(val_loader)}")

    x, y = next(iter(train_loader))

    print("Input shape :", x.shape)
    print("Target shape:", y.shape)

    decoded_input = tokenizer.decode(x[0], skip_special_tokens=False)

    target_tokens = y[0][y[0] != -100]
    decoded_target = tokenizer.decode(target_tokens, skip_special_tokens=False)

    print("\nDecoded input:")
    print(decoded_input)

    print("\nDecoded target / loss computed part:")
    print(decoded_target)