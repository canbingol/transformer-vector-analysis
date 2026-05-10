# Vector Alignment Analysis

This repository contains the experimental setup prepared for an Erasmus course essay titled **"Vector-Based Analysis of Alignment in Large Language Models"**.

The experiments study how vector representations in language models change during pretraining and instruction tuning. The goal is to support the essay with small-scale, inspectable experiments using embedding spaces, attention behavior, a decoder-only language model, and comparisons between pretrained and instruction-tuned model outputs.

## Contents

- `01_embedding/`: Word2Vec embedding experiments and embedding analysis.
- `02_attn_heatmap/`: Attention heatmap visualizations.
- `03_pretrain/`: Decoder model pretraining setup using the TinyStories dataset.
- `04_instruction_tuning/`: Instruction tuning setup using the OpenHermes-2.5 dataset.
- `essay.md`: Essay draft titled **"Vector-Based Analysis of Alignment in Large Language Models"**.
- `checkpoints/`: Local folder for model checkpoint files.

## Model Checkpoints

The trained and random model checkpoints are hosted separately on Hugging Face:

https://huggingface.co/canbingol/transformer-vector-analysis-checkpoints

After downloading the checkpoint files, place them in the `checkpoints/` folder. The hosted checkpoint files are:

- `random_decoder.pt`
- `pretrained_decoder.pt`
- `openhermes_instruction_tuned_decoder.pt`
- `alpaca_instruction_tuned_decoder.pt`

The essay uses `openhermes_instruction_tuned_decoder.pt` for the final instruction-tuned model analysis.

## Setup

Install the core training dependencies in your Python environment:

```bash
pip install torch transformers datasets tqdm numpy
```

For the analysis notebooks, you may also need:

```bash
pip install matplotlib seaborn pandas scikit-learn gensim nltk
```

The code can run on GPU, Apple Silicon MPS, or CPU. The training scripts try to select the available device automatically.

## Usage

The `03_pretrain/` folder can be used to pretrain the decoder model. First prepare the pretraining data:

```bash
cd 03_pretrain
python prepare_dataset.py
```

Then start pretraining:

```bash
python trainer.py
```

The `04_instruction_tuning/` folder can be used for instruction tuning. First make sure `checkpoints/pretrained_decoder.pt` exists. Then run:

```bash
cd ../04_instruction_tuning
python prepare_dataset.py
python trainer.py
```

Running `04_instruction_tuning/trainer.py` saves a newly trained checkpoint as `checkpoints/instruction_tuned_decoder.pt` by default. The hosted OpenHermes checkpoint used for the essay is named `openhermes_instruction_tuned_decoder.pt`.

Analysis and visualization notebooks are available in the related folders:

- `01_embedding/embedding.ipynb`
- `02_attn_heatmap/heatmap.ipynb`
- `03_pretrain/analysis.ipynb`
- `04_instruction_tuning/analysis.ipynb`

## Datasets

The project uses two main datasets:

- Pretraining: `roneneldan/TinyStories`
- Instruction tuning: `teknium/OpenHermes-2.5`

Datasets are downloaded from Hugging Face through the `datasets` library.

## Essay Model Choice

The model used in the essay is the instruction-tuned model trained with the OpenHermes dataset. An Alpaca-based version was also tested, but it was trained with less data, so the OpenHermes version was selected for the final essay analysis.

## Notes

This repository is intended for research and learning in the context of the essay. The model architecture is intentionally small; the main focus is not large-scale performance, but observing changes in vector representations and experimentally analyzing the alignment process.
