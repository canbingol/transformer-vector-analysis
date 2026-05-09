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

            skipped_long = 0
            skipped_invalid = 0

            for conversations in tqdm(hf_dataset["conversations"]):
                result = self.convert_conversation(conversations)

                if result is None:
                    skipped_invalid += 1
                    continue

                prompt, answer = result

                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False
                )["input_ids"]

                answer_ids = tokenizer(
                    answer,
                    add_special_tokens=False
                )["input_ids"]

                full_ids = prompt_ids + answer_ids

                # input_ids = full_ids[:-1] olduğu için,
                # input uzunluğu max_seq_len'i geçmemeli.
                if len(full_ids) - 1 > max_seq_len:
                    skipped_long += 1
                    continue

                input_ids = full_ids[:-1]
                target_ids = [-100] * (len(prompt_ids) - 1) + answer_ids

                if len(input_ids) < max_seq_len:
                    pad_len = max_seq_len - len(input_ids)

                    input_ids += [pad_token_id] * pad_len
                    target_ids += [-100] * pad_len

                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_ids, dtype=torch.long))

            print(f"Loaded examples: {len(self.input_ids)}")
            print(f"Skipped long examples: {skipped_long}")
            print(f"Skipped invalid examples: {skipped_invalid}")

        def convert_conversation(self, conversations):
            human_message = None
            assistant_message = None

            for message in conversations:
                role = message.get("from")
                value = message.get("value")

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

    return train_dataloader, val_dataloader, tokenizer


if __name__ == "__main__":

    train_loader, val_loader, tokenizer = prepare_it_data(
        batch_size=4,
        max_seq_len=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False
    )

    print(f"len train loader: {train_loader}")
    print(f"len val loader: {val_loader}")

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