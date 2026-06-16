# 1LLM2RuleThemAll

A hands-on learning project for understanding how large language models work by building a small GPT-style model from scratch and training it on text data.

## Overview

This repository combines:
- model architecture code for a transformer-based language model
- text preprocessing utilities
- training and generation scripts
- notebooks for experimenting with data and model behavior

The main goal is to learn how LLMs work in practice, not to build a production-ready model.

## Project Goals

- Understand the basics of tokenization, embeddings, and positional encoding
- Explore transformer blocks and self-attention
- Train a small model for next-token prediction
- Experiment with text generation and observe model behavior
- Build intuition about how data and architecture affect results

## Repository Structure

### LLMfromScratch/
Contains the core model implementation and training logic.
- `architecture.py` — defines the GPT-style model and transformer blocks
- `attention.py` — multi-head attention implementation
- `train.py` — training loop, loss evaluation, and generation helpers
- `experiment.py` — script used to run training experiments
- `utils.py` — additional utility functions for generation

### TextDataProcessing/
Contains code related to preparing and handling text data.
- `data.py` — dataset and dataloader setup for training
- `DataPipeLine.ipynb` — notebook for processing text data
- `Pdf2TextFileConvert.ipynb` — notebook for converting PDFs into text files

### BookAndDataFiles/
Stores the dataset used for training.
- `txtFile/book.txt` — text corpus used by the project

## Dataset

The project uses text from The Lord of the Rings series as the training corpus.
The data is tokenized and split into overlapping chunks so the model can learn patterns from sequential text.

## Model Design

The model is a decoder-style transformer inspired by GPT, with:
- token embeddings
- positional embeddings
- multi-head self-attention
- feed-forward layers
- normalization and dropout

The training objective is next-token prediction.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure the training text file is available in the expected location.
3. If needed, run the preprocessing notebooks before training.

## How to Run

From the project root, run:

```bash
python LLMfromScratch/experiment.py
```

This script loads the text data, builds the model, trains it, and prints sample generated text during training.

## Expected Outputs

Depending on the run, you can expect:
- training and validation loss values
- sample generated text during training
- saved model artifacts after the experiment

## Notes

- This is a learning and experimentation project.
- The focus is on understanding model behavior and training flow.
- Results may vary depending on hardware, dataset size, and hyperparameters.

## Next Steps

Potential improvements for future experiments:
- try different model sizes
- adjust context length and batch size
- experiment with more data or better preprocessing
- compare performance across hyperparameter settings


