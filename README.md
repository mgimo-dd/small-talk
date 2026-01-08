# small-talk

A tiny LLM from HF that sustains some small talk

This project uses the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model from Hugging Face to enable simple conversational AI capabilities.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mgimo-dd/small-talk.git
cd small-talk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the chat script:
```bash
python chat.py
```

The script will load the Qwen3-0.6B model and generate a response to a sample prompt.

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- Transformers 4.37.0+
- Accelerate 0.26.0+

See `requirements.txt` for full dependency list.

## Model Information

This project uses Qwen3-0.6B, a compact language model created by Alibaba Cloud. For more information, visit the [model page on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B).
