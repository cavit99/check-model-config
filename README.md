# check-model-config

A validation tool for transformer model configurations that helps identify inconsistencies between model configs, weights, and tokenizers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

`check-model-config` is a CLI tool that performs comprehensive validation checks on transformer models to ensure their configurations are consistent with their weights and tokenizers. It helps identify potential issues that could affect model performance or behavior.

The tool is particularly useful for:
- Model developers who want to ensure their published models have consistent configurations
- Users who want to validate models before using them in production
- Researchers working with modified or custom models who need to verify configuration integrity

## Installation

```bash
pip install check-model-config
```

## Usage

### Basic Usage

Run the validation tests on a model:

```bash
check-model-config --model openai/gpt2
```

You can use either a Hugging Face model ID or a local directory path:

```bash
check-model-config --model ./path/to/local/model
```

### Skip Weight-Dependent Tests

For faster validation or when working with very large models, you can skip tests that require loading model weights:

```bash
check-model-config --model meta-llama/Llama-2-70b --load-weights=False
```

## Validation Tests

The tool runs the following validation checks:

- **Vocabulary Size Consistency**: Verifies that vocab_size in config matches embedding weights, lm_head weights, and tokenizer
- **Hidden Size Consistency**: Ensures hidden dimensions match across weights and config
- **Layer Count Validation**: Checks that layer counts in config match actual weights
- **Attention Mechanism**: Validates KQV projections and attention output dimensions
- **MLP Layers**: Checks feed-forward network dimensions and activation functions
- **Tied Embeddings**: Verifies embedding tying configuration matches implementation
- **Position Embeddings**: Validates position embedding configuration and implementation
- **Mixture of Experts**: Checks MoE configurations for models that use experts
- **Window Layers**: Validates sliding window attention configurations

## Requirements

- Python 3.9+
- PyTorch 2.5.1+
- Transformers 4.47.1+
- Accelerate 1.2.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
