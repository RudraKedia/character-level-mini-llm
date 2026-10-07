# Character-Level Mini LLM

A small character-level language model built from scratch using **PyTorch**.

This project was created to understand the internal working of a Transformer-based language model by implementing its major components rather than using a pretrained LLM.

The model is trained on *The Merchant of Venice* by William Shakespeare and learns to predict the next character based on the characters that came before it.

## Project Overview

The overall pipeline of the project is:

```text
The Merchant of Venice
        ↓
Character Tokenization
        ↓
Train / Validation Split
        ↓
Token Embeddings
        ↓
Transformer Blocks
        ↓
GQA Attention + RoPE
        ↓
SwiGLU Feed-Forward Network
        ↓
Final RMSNorm
        ↓
Language Model Head
        ↓
Next Character Prediction
        ↓
Text Generation
```

## Features

- Character-level tokenization
- Token embeddings
- Causal self-attention
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward network
- Pre-Norm Transformer architecture
- Residual connections
- Weight tying
- Cross-entropy loss
- AdamW optimizer
- Gradient clipping
- Validation loss evaluation
- Temperature-based text generation
- Interactive user-provided prompts

## Dataset

The model is trained on **The Merchant of Venice** by William Shakespeare.

The text is processed at the character level, meaning individual characters are treated as tokens instead of words or subwords.

### Dataset Size

| Split | Characters |
|---|---:|
| Total | 130,853 |
| Training | 111,225 |
| Validation | 19,628 |

The dataset is split approximately **90% for training** and **10% for validation**.

## Model Architecture

### Token Embedding

Each character is converted into an integer token ID and then mapped to a learnable vector representation.

```text
Character ID
     ↓
Embedding Table
     ↓
Vector Representation
```

The model uses a `d_model` of **256**, meaning each character is represented by a 256-dimensional vector.

### Grouped Query Attention

The attention mechanism uses:

- **8 Query heads**
- **2 Key/Value heads**

Multiple Query heads share Key/Value heads, reducing Key/Value computation and memory usage.

### Rotary Positional Embeddings

The model uses **RoPE** to provide positional information to the attention mechanism.

Instead of adding positional embeddings to token embeddings, RoPE applies position-dependent rotations to Query and Key representations.

### RMSNorm

The Transformer uses **RMSNorm** instead of LayerNorm.

RMSNorm normalizes representations using their root mean square and is applied in a Pre-Norm Transformer architecture.

### SwiGLU

The feed-forward network uses **SwiGLU**.

It uses a SiLU-based gating mechanism to control information flow through the feed-forward network.

### Residual Connections

Residual connections allow the original representation to be passed forward along with the output of the attention and feed-forward sublayers.

The Transformer block follows:

```text
x
 ↓
RMSNorm
 ↓
GQA Attention
 ↓
+ Residual
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
+ Residual
 ↓
Output
```

## Model Configuration

The model was trained with the following configuration:

| Parameter | Value |
|---|---:|
| `d_model` | 256 |
| Transformer layers | 4 |
| Attention heads | 8 |
| KV heads | 2 |
| FFN hidden dimension | 680 |
| Maximum context length | 256 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Training steps | 3000 |

## Training Objective

The model is trained using **next-character prediction**.

For example:

```text
Input:
HELLO

Target:
ELLO
```

At every position, the model predicts the character that should come next.

The training process follows:

```text
Input Batch
    ↓
Forward Pass
    ↓
Calculate Cross-Entropy Loss
    ↓
Clear Gradients
    ↓
Backpropagation
    ↓
Gradient Clipping
    ↓
Update Parameters
    ↓
Repeat
```

## Text Generation

After training, the model can generate text from a user-provided prompt.

For example:

```text
Enter your prompt: ANTONIO:
```

The model then generates a continuation character by character.

Generation is autoregressive:

```text
Prompt
  ↓
Predict next character
  ↓
Append character
  ↓
Use updated sequence
  ↓
Predict next character
  ↓
Repeat
```

### Temperature

Temperature controls the randomness of generation.

| Temperature | Behavior |
|---:|---|
| `0.5` | More conservative and predictable |
| `0.8` | Balanced |
| `1.0` | More random |
| `1.2` | More creative but potentially less coherent |

For example, prompts such as:

```text
ANTONIO:
PORTIA:
SHYLOCK:
BASSANIO:
```

can be used to generate different continuations.

The generated text is produced from learned character-level patterns and is not simply retrieved from the original text.

## Results

The model successfully learned patterns from the training data, including:

- Shakespearean vocabulary
- Character names
- Dialogue formatting
- Punctuation
- Common character sequences
- Basic grammatical patterns

Example generation:

```text
ANTONIO:
In sooth, I know not why I am so sad.
It wearies me; you say it wearies you;
```

Because this is a small educational model trained on a limited dataset, generated text can still contain:

- Repetition
- Grammatical errors
- Unusual character combinations
- Incoherent passages

## Technologies Used

- **Python**
- **PyTorch**
- **Jupyter Notebook**
- **CUDA**
- **NVIDIA GPU**

Development hardware:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
PyTorch: 2.11.0+cu128
CUDA: 12.8
```

## Repository Structure

Currently, the project is intentionally simple:

```text
character-level-mini-llm/
│
├── first_model.ipynb
└── README.md
```

The complete implementation, training process, evaluation, and generation code are contained in `first_model.ipynb`.

## What I Learned

This project helped me understand the practical implementation of:

- Character-level tokenization
- Token embeddings
- Tensor shapes and transformations
- Query, Key, and Value
- Self-attention
- Causal masking
- Grouped Query Attention
- Rotary Positional Embeddings
- RMSNorm
- SwiGLU
- Transformer blocks
- Residual connections
- Weight tying
- Cross-entropy loss
- Backpropagation
- AdamW
- Gradient clipping
- Training and evaluation modes
- Autoregressive generation
- Temperature sampling
- GPU acceleration with PyTorch

## Limitations

This is a **small educational language model**, not a production-scale LLM.

The main limitations are:

- Character-level tokenization is less efficient than subword tokenization.
- The training dataset is relatively small.
- The model has a small parameter count.
- The model can overfit the training data.
- Generated text is not always coherent.
- The model only learns patterns available in its training corpus.

The primary purpose of this project is to understand how Transformer-based language models work internally.

## Future Improvements

Possible improvements include:

- Implementing BPE or another subword tokenizer
- Training on a larger corpus
- Improving regularization and generalization
- Saving and loading the best validation checkpoint
- Experimenting with larger model configurations
- Adding top-k sampling
- Adding top-p (nucleus) sampling
- Experimenting with different context lengths
- Comparing character-level and subword tokenization
- Building a simple interface for text generation

## Project Goal

The main goal of this project was not to build a large-scale or production-ready LLM.

It was to understand the internal pipeline of a Transformer language model by implementing its major components and training a small model from scratch.

> **Characters → Embeddings → Attention → Transformer Blocks → Logits → Generated Text**

## Author

**Rudra Kedia**

B.Tech — Information Technology  
IIIT Bhubaneswar
