# AI-Powered Grief Support Message Classification System

A robust, multi-model system designed to analyze and classify text messages in grief support contexts. This system evaluates messages for sentiment, response necessity, and potential crisis situations, helping support teams prioritize and respond appropriately.

## System Overview

This system utilizes three distinct AI models to classify text messages from individuals experiencing grief:

1. **RoBERTa-base (125M)** - An encoder-only model with task-specific classification heads
2. **LLM2Vec (8B)** - A converted encoder model based on Llama 3.1 with frozen weights
3. **Llama 3.1 Instruct (8B, 4-bit quantized)** - A decoder model using few-shot prompting

Each model is designed to classify messages across three dimensions:
- **Sentiment** (Positive, Neutral, Negative)
- **Response Needed** (Yes, No)
- **Crisis Detection** (Crisis, No Crisis)

The system is built to run entirely on-premise, without relying on external APIs or cloud services, and uses only open-source models.

## Architecture

### Multitask Classification Architecture

The encoder models (RoBERTa and LLM2Vec) use a shared architecture with specialized classification heads:

![Multitask Classification Model Architecture](./images/classification.pdf)
*Figure 1: Multitask Classification Model Architecture with shared encoder and separate classification heads.*

### Prompt-Based Approach

The Llama 3.1 model uses a different approach based on few-shot learning and prompt engineering:

![Prompt-based approach](./images/prompt.pdf)
*Figure 2: Prompt-based approach using Llama 3.1 8B 4-bit.*

## Key Features

- **Conversation Context**: Incorporates previous message history for more accurate classification
- **Attention Mechanism**: Dynamically weighs the importance of historical context vs. current message
- **Crisis-First Design**: Optimized for high recall in crisis detection to minimize missed cases
- **Multi-Model Support**: Choose between different models based on your computational resources and needs
- **Explainable Results**: Provides confidence scores and reasoning (Llama 3.1 model)
- **Interactive Interface**: Easy-to-use Gradio web interface for evaluating messages

## Dataset

The system was trained on a synthetic dataset of 2,000 examples generated using GPT-4o. The dataset includes diverse grief scenarios with balanced distributions:

- Sentiment: 33% positive, 33% neutral, 33% negative
- Response Needed: 50% requiring response, 50% not requiring
- Crisis Detection: 10-15% crisis situations, 85-90% non-crisis

The dataset includes ambiguous cases and covers various grief contexts (recent loss, anniversary grief, terminal illness, etc.) to ensure robust performance.

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch
- Gradio
- LangChain
- Ollama (for the RAG/Llama model)

### Dependencies

```bash
pip install torch gradio langchain langchain-community
```

For the RAG model, you'll need Ollama installed and running with the Llama 3.1 model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.1:latest
```

### Model Files

You'll need the following model files:
- `grief_model_v3/` - RoBERTa model files
- `grief_model_llm2vec_v2/` - LLM2Vec model files

## Usage

1. Save the provided interface code as `grief_eval.py`
2. Run the application:
```bash
python grief_eval.py
```
3. Access the web interface (by default at http://localhost:7860)
4. Select a model, choose or enter a scenario, and evaluate responses

## Interface Guide

The Gradio interface allows you to:
- Select between RoBERTa, LLM2Vec, or Llama 3.1 (RAG) models
- Load predefined scenarios or create your own
- Provide conversation history for context
- Enter support messages and recipient replies
- View detailed classification results with confidence scores

Example scenarios are provided, including general grief support, recent loss, and potentially concerning situations.

## Model Performance

| Metric | RoBERTa | LLM2Vec | Llama 3.1 Prompt |
|--------|---------|---------|------------------|
| Sentiment F1 | 0.82 | 0.63 | 0.72 |
| Response F1 | 0.89 | 0.77 | 0.89 |
| Crisis Recall | 0.88 | 0.78 | 0.91 |

*Note: LLM2Vec is underfit as only the classification heads were trained, not the full model.*

## Model Selection Considerations

- **RoBERTa**: Fastest inference (400ms), lowest memory requirements, best sentiment classification
- **LLM2Vec**: Higher quality embeddings but requires more memory and training
- **Llama 3.1**: Best crisis recall, provides reasoning for classifications, no training required

## Ethical Considerations

This system is designed with several ethical safeguards:
- Prioritizes recall over precision for crisis detection
- Uses conservative thresholds for flagging potential crisis situations
- Provides explainable outputs to aid human reviewers
- Can handle linguistic and cultural variations
