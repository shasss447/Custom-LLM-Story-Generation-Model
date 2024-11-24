# Custom LLM: Story Generation Model 

## Project Overview

This project implements a custom large language model (LLM) from scratch using `torch` for story generation, trained on a corpus of stories sourced from Wikisource. The architecture is inspired by `GPT`, leveraging a transformer-only decoder model and causal multi-head attention for autoregressive text generation.

## Features

- **Transformer Decoder-Only Architecture**: Implements causal attention for autoregressive sequence modeling.
- **GPT-2 Tokenizer**: Utilizes GPT-2’s pre-trained tokenizer for efficient tokenization and decoding.
- **Custom Multi-Head Causal Attention**: Scratch implementation of causal attention for text generation.
- **Flexible Text Generation**: Supports generation with adjustable parameters like temperature and top-k sampling.

## Model Architecture

- The model employs a transformer decoder-only architecture, following the design principles of GPT.
- Each transformer block includes:
   - Multi-Head Causal Attention
   - Layer Normalization
   - Feedforward Networks with *GELU* Activation

### Results
Below is the graph showing the loss trends for both training and validation

![](/plot.png)