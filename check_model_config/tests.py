import gc
import json
import os
import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

def get_model_setup(model_path):
    """Load config, model, and tokenizer for a given model path."""
    if model_path is None:
        raise ValueError("Model path must be provided via CHECK_MODEL_PATH environment variable")
    
    device = "auto" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    if Path(model_path).is_dir():
        with open(Path(model_path) / "config.json", "r") as f:
            raw_config = json.load(f)
    else:
        raw_config = PretrainedConfig.from_pretrained(model_path).to_dict()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    config = model.config
    print(f"Model loaded on device: {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    print("Model Config:", config.to_dict())

    return {
        "model": model,
        "config": config,
        "raw_config": raw_config,
        "tokenizer": tokenizer,
        "device": device
    }

@pytest.fixture(scope="module")
def model_setup():
    """Fixture to provide model setup with dynamic model path."""
    model_path = os.environ.get("CHECK_MODEL_PATH")
    if not model_path:
        raise ValueError("CHECK_MODEL_PATH environment variable not set. Run with 'check-model-config --model <path>'")
    setup = get_model_setup(model_path)
    yield setup
    
    # Cleanup
    del setup["model"]
    del setup["tokenizer"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        torch.mps.synchronize()
    gc.collect()

def test_config_vocab_size_vs_weights_and_tokenizer(model_setup):
    """Verify config vocab_size matches embedding weights and tokenizer."""
    config_vocab_size = model_setup["raw_config"]["vocab_size"]
    embed_weight_shape = list(model_setup["model"].get_input_embeddings().weight.shape)
    actual_vocab_size = embed_weight_shape[0]
    tokenizer_vocab_size = len(model_setup["tokenizer"])
    
    assert config_vocab_size == actual_vocab_size, (
        f"Config vocab_size ({config_vocab_size}) does not match embedding weight size ({actual_vocab_size})"
    )
    assert config_vocab_size >= tokenizer_vocab_size, (
        f"Config vocab_size ({config_vocab_size}) must be >= tokenizer vocab size ({tokenizer_vocab_size})"
    )
    print(f"Config vocab_size ({config_vocab_size}) matches embedding weight size ✓")
    if config_vocab_size > tokenizer_vocab_size:
        print(f"Note: Config vocab_size > tokenizer vocab size ({config_vocab_size} vs {tokenizer_vocab_size}) - likely padding tokens")
        pytest.warns(
            UserWarning,
            match=f"Config vocab_size > tokenizer vocab size ({config_vocab_size} vs {tokenizer_vocab_size})"
        )
    else:
        print(f"Config vocab_size matches tokenizer vocab size ✓")

def test_config_hidden_size_vs_weights(model_setup):
    """Verify config hidden_size matches embedding and attention weights."""
    config_hidden_size = model_setup["raw_config"]["hidden_size"]
    embed_hidden_size = model_setup["model"].get_input_embeddings().weight.shape[1]
    assert config_hidden_size == embed_hidden_size, (
        f"Config hidden_size ({config_hidden_size}) does not match embedding hidden size ({embed_hidden_size})"
    )
    print(f"Config hidden_size ({config_hidden_size}) matches embedding weights ✓")

def test_num_layers(model_setup):
    """Test the number of hidden layers."""
    config_num_layers = model_setup["raw_config"]["num_hidden_layers"]
    actual_num_layers = len(model_setup["model"].model.layers)
    assert config_num_layers == actual_num_layers, (
        f"Config num_hidden_layers ({config_num_layers}) does not match actual layers ({actual_num_layers})"
    )
    print(f"Number of layers: {actual_num_layers} ✓")

@pytest.mark.parametrize("layer_idx", [0, lambda x: int(x["config"].num_hidden_layers / 2), lambda x: x["config"].num_hidden_layers - 1])
def test_attention_mechanism(model_setup, layer_idx):
    """Detailed check of attention heads and weights across layers."""
    if callable(layer_idx):
        layer_idx = layer_idx(model_setup)
    config_num_heads = model_setup["raw_config"]["num_attention_heads"]
    config_kv_heads = model_setup["raw_config"].get("num_key_value_heads", config_num_heads)
    hidden_size = model_setup["raw_config"]["hidden_size"]
    head_dim = hidden_size // config_num_heads
    expected_kv_dim = config_kv_heads * head_dim

    layer = model_setup["model"].model.layers[layer_idx].self_attn
    q_shape = list(layer.q_proj.weight.shape)
    k_shape = list(layer.k_proj.weight.shape)
    assert q_shape == [hidden_size, hidden_size], (
        f"Layer {layer_idx} q_proj mismatch: expected [{hidden_size}, {hidden_size}], got {q_shape}"
    )
    assert k_shape == [expected_kv_dim, hidden_size], (
        f"Layer {layer_idx} k_proj mismatch: expected [{expected_kv_dim}, {hidden_size}], got {k_shape}"
    )
    print(f"Layer {layer_idx} q_proj shape: {q_shape} ✓")
    print(f"Layer {layer_idx} k_proj shape: {k_shape} ✓")

@pytest.mark.parametrize("layer_idx", [0, lambda x: int(x["config"].num_hidden_layers / 2), lambda x: x["config"].num_hidden_layers - 1])
def test_mlp_layers(model_setup, layer_idx):
    """Test intermediate size in MLP/FFN layers across specific indices."""
    if callable(layer_idx):
        layer_idx = layer_idx(model_setup)
    config_intermediate_size = model_setup["raw_config"]["intermediate_size"]
    hidden_size = model_setup["raw_config"]["hidden_size"]
    layer = model_setup["model"].model.layers[layer_idx].mlp
    if hasattr(layer, "gate_proj"):
        gate_shape = list(layer.gate_proj.weight.shape)
        assert gate_shape == [config_intermediate_size, hidden_size], (
            f"Layer {layer_idx} MLP gate_proj mismatch: expected [{config_intermediate_size}, {hidden_size}], got {gate_shape}"
        )
        print(f"Layer {layer_idx} MLP gate_proj shape: {gate_shape} ✓")
    else:
        print(f"Layer {layer_idx} has no gate_proj; skipping MLP check")

def test_tied_embeddings(model_setup):
    """Verify tied embeddings configuration."""
    tie_word_embeddings = model_setup["raw_config"].get("tie_word_embeddings", False)
    if not tie_word_embeddings:
        assert hasattr(model_setup["model"], "lm_head"), (
            "Expected separate lm_head for untied embeddings, but none found"
        )
        lm_shape = list(model_setup["model"].lm_head.weight.shape)
        expected_shape = [model_setup["raw_config"]["vocab_size"], model_setup["raw_config"]["hidden_size"]]
        assert lm_shape == expected_shape, (
            f"Output embedding mismatch: expected {expected_shape}, got {lm_shape}"
        )
        print(f"Output embedding shape: {lm_shape} ✓")
    else:
        assert not (hasattr(model_setup["model"], "lm_head") and model_setup["model"].lm_head is not None), (
            "lm_head should not exist with tied embeddings"
        )
        print("Word embeddings tied, no separate lm_head ✓")

def test_position_embeddings(model_setup):
    """Check max_position_embeddings and detect RoPE."""
    max_pos = model_setup["raw_config"].get("max_position_embeddings", 2048)  # Default if not specified
    assert max_pos > 0, f"max_position_embeddings must be positive, got {max_pos}"
    print(f"max_position_embeddings: {max_pos} ✓")
    rope_found = hasattr(model_setup["model"].model.layers[0].self_attn, "rotary_emb") or "rope" in str(model_setup["model"].model.layers[0].self_attn).lower()
    if not rope_found:
        print("Warning: No clear RoPE implementation found; assuming max_position_embeddings is valid")
        pytest.warns(
            UserWarning,
            match="No clear RoPE implementation found"
        )
    else:
        print("RoPE implementation confirmed ✓")

def test_window_layers(model_setup):
    """Validate sliding window settings."""
    max_window = model_setup["raw_config"].get("max_window_layers", model_setup["raw_config"]["num_hidden_layers"])
    num_layers = model_setup["raw_config"]["num_hidden_layers"]
    use_sliding_window = model_setup["raw_config"].get("use_sliding_window", False)
    if not use_sliding_window:
        print("Sliding window disabled - max_window_layers is ignored unless enabled")
    else:
        assert max_window <= num_layers, (
            f"Config Error: max_window_layers ({max_window}) exceeds num_hidden_layers ({num_layers})"
        )
        print(f"Sliding window enabled - max_window_layers ({max_window}) ≤ num_hidden_layers ({num_layers}) ✓")

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])