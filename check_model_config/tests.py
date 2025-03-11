import gc
import json
import os
import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

# Check if weights should be loaded (set by cli.py)
load_weights = os.environ.get("CHECK_LOAD_WEIGHTS", "True") == "True"

def get_model_setup(model_path, load_weights=True):
    """Load config, and optionally model and tokenizer, based on load_weights flag."""
    if model_path is None:
        raise ValueError("Model path must be provided via CHECK_MODEL_PATH environment variable")
    
    # Load raw config
    if Path(model_path).is_dir():
        with open(Path(model_path) / "config.json", "r") as f:
            raw_config = json.load(f)
    else:
        raw_config = PretrainedConfig.from_pretrained(model_path).to_dict()

    setup = {"raw_config": raw_config}
    
    if load_weights:
        device = "auto" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        setup["model"] = model
        setup["config"] = model.config
        setup["device"] = device
        print(f"Model loaded on device: {next(model.parameters()).device}")
    else:
        setup["config"] = PretrainedConfig.from_pretrained(model_path)
        setup["device"] = "none"
        print("Skipping model weight loading")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    setup["tokenizer"] = tokenizer
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    print("Model Config:", setup["config"].to_dict())

    return setup

@pytest.fixture(scope="module")
def model_setup():
    """Fixture to provide model setup with dynamic model path and load_weights flag."""
    model_path = os.environ.get("CHECK_MODEL_PATH")
    load_weights_flag = os.environ.get("CHECK_LOAD_WEIGHTS", "True") == "True"
    if not model_path:
        raise ValueError("CHECK_MODEL_PATH environment variable not set. Run with 'check-model-config --model <path>'")
    setup = get_model_setup(model_path, load_weights_flag)
    yield setup
    
    # Cleanup
    if load_weights_flag:
        del setup["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            torch.mps.synchronize()
    del setup["tokenizer"]
    gc.collect()

@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_config_vocab_size_vs_weights_and_tokenizer(model_setup):
    """Verify config vocab_size matches embedding weights, output weights, and tokenizer."""
    config = model_setup["config"]
    config_vocab_size = config.vocab_size
    embed_weight_shape = list(model_setup["model"].get_input_embeddings().weight.shape)
    actual_vocab_size = embed_weight_shape[0]
    tokenizer_vocab_size = len(model_setup["tokenizer"])
    
    assert config_vocab_size == actual_vocab_size, (
        f"Config vocab_size ({config_vocab_size}) does not match embedding weight size ({actual_vocab_size})"
    )
    if hasattr(model_setup["model"], "lm_head"):
        lm_head_shape = list(model_setup["model"].lm_head.weight.shape)
        assert config_vocab_size == lm_head_shape[0], (
            f"Config vocab_size ({config_vocab_size}) does not match lm_head weight size ({lm_head_shape[0]})"
        )
    assert config_vocab_size >= tokenizer_vocab_size, (
        f"Config vocab_size ({config_vocab_size}) must be >= tokenizer vocab size ({tokenizer_vocab_size})"
    )
    print(f"Config vocab_size ({config_vocab_size}) matches embedding weight size ✓")
    if hasattr(model_setup["model"], "lm_head"):
        print(f"Config vocab_size ({config_vocab_size}) matches lm_head weight size ✓")
    if config_vocab_size > tokenizer_vocab_size:
        print(f"Note: Config vocab_size > tokenizer vocab size ({config_vocab_size} vs {tokenizer_vocab_size}) - likely padding tokens")
        pytest.warns(
            UserWarning,
            match=f"Config vocab_size > tokenizer vocab size ({config_vocab_size} vs {tokenizer_vocab_size})"
        )
    else:
        print(f"Config vocab_size matches tokenizer vocab size ✓")

@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_config_hidden_size_vs_weights(model_setup):
    """Verify config hidden_size matches embedding and attention weights."""
    config = model_setup["config"]
    config_hidden_size = config.hidden_size
    embed_hidden_size = model_setup["model"].get_input_embeddings().weight.shape[1]
    assert config_hidden_size == embed_hidden_size, (
        f"Config hidden_size ({config_hidden_size}) does not match embedding hidden size ({embed_hidden_size})"
    )
    print(f"Config hidden_size ({config_hidden_size}) matches embedding weights ✓")

def test_num_layers_config(model_setup):
    """Test the number of hidden layers in the configuration."""
    config = model_setup["config"]
    config_num_layers = config.num_hidden_layers
    assert config_num_layers > 0, f"num_hidden_layers must be positive, got {config_num_layers}"
    print(f"Config num_hidden_layers: {config_num_layers} ✓")

def test_raw_config_vs_config(model_setup):
    """Ensure raw config.json matches the processed config, warn on discrepancies."""
    raw_config = model_setup["raw_config"]
    config = model_setup["config"]
    raw_dict = raw_config
    config_dict = config.to_dict()
    differences = {k: (raw_dict.get(k), config_dict[k]) for k in set(raw_dict) | set(config_dict) if raw_dict.get(k) != config_dict.get(k)}
    defaults_injected = {k: config_dict[k] for k in config_dict if k not in raw_dict}
    if differences:
        print(f"Warning: Raw config.json differs from processed config: {differences}")
        pytest.warns(
            UserWarning,
            match=f"Raw config.json differs from processed config: {differences}"
        )
    if defaults_injected:
        print(f"Warning: Fields injected by PretrainedConfig: {defaults_injected}")
        pytest.warns(
            UserWarning,
            match=f"PretrainedConfig injected defaults not in config.json: {defaults_injected}"
        )
    print("Raw config.json vs processed config checked ✓")

def test_required_fields(model_setup):
    """Check for required fields based on model architecture."""
    config = model_setup["config"]
    model_type = getattr(config, "model_type", "unknown").lower()
    REQUIRED_FIELDS = {
        "qwen": ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"],
        "llama": ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"],
        "mixtral": ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_local_experts"],
        "deepseek": ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "n_routed_experts"],
    }
    required = REQUIRED_FIELDS.get(model_type, ["vocab_size", "hidden_size", "num_hidden_layers"])
    for field in required:
        assert hasattr(config, field), f"Missing required field for {model_type}: {field}"
        assert getattr(config, field) is not None, f"Required field {field} is None for {model_type}"
    print(f"Required fields for {model_type} present: {required} ✓")

@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_num_layers_weights(model_setup):
    """Test the number of hidden layers against actual model weights."""
    config = model_setup["config"]
    config_num_layers = config.num_hidden_layers
    actual_num_layers = len(model_setup["model"].model.layers)
    assert config_num_layers == actual_num_layers, (
        f"Config num_hidden_layers ({config_num_layers}) does not match actual layers ({actual_num_layers})"
    )
    print(f"Number of layers matches weights: {actual_num_layers} ✓")

@pytest.mark.parametrize("layer_idx", [0, lambda x: int(x["config"].num_hidden_layers / 2), lambda x: x["config"].num_hidden_layers - 1])
@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_attention_mechanism(model_setup, layer_idx):
    """Detailed check of attention heads and weights across layers."""
    if callable(layer_idx):
        layer_idx = layer_idx(model_setup)
    config = model_setup["config"]
    config_num_heads = config.num_attention_heads
    config_kv_heads = getattr(config, "num_key_value_heads", config_num_heads)
    hidden_size = config.hidden_size
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
@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_mlp_layers(model_setup, layer_idx):
    """Test intermediate size in MLP/FFN layers across specific indices."""
    if callable(layer_idx):
        layer_idx = layer_idx(model_setup)
    config = model_setup["config"]
    config_intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    layer = model_setup["model"].model.layers[layer_idx].mlp
    if hasattr(layer, "gate_proj"):
        gate_shape = list(layer.gate_proj.weight.shape)
        assert gate_shape == [config_intermediate_size, hidden_size], (
            f"Layer {layer_idx} MLP gate_proj mismatch: expected [{config_intermediate_size}, {hidden_size}], got {gate_shape}"
        )
        print(f"Layer {layer_idx} MLP gate_proj shape: {gate_shape} ✓")
    else:
        print(f"Layer {layer_idx} has no gate_proj; skipping MLP check")

@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_tied_embeddings(model_setup):
    """Verify tied embeddings configuration."""
    config = model_setup["config"]
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    if not tie_word_embeddings:
        assert hasattr(model_setup["model"], "lm_head"), (
            "Expected separate lm_head for untied embeddings, but none found"
        )
        lm_shape = list(model_setup["model"].lm_head.weight.shape)
        expected_shape = [config.vocab_size, config.hidden_size]
        assert lm_shape == expected_shape, (
            f"Output embedding mismatch: expected {expected_shape}, got {lm_shape}"
        )
        print(f"Output embedding shape: {lm_shape} ✓")
    else:
        assert not (hasattr(model_setup["model"], "lm_head") and model_setup["model"].lm_head is not None), (
            "lm_head should not exist with tied embeddings"
        )
        print("Word embeddings tied, no separate lm_head ✓")

@pytest.mark.skipif(not load_weights, reason="Requires model weights")
def test_moe_config(model_setup):
    """Validate Mixture of Experts (MoE) configuration and weights."""
    model = model_setup["model"]
    config = model_setup["config"]
    # Test a subset of layers: first, middle, last
    for layer_idx in [0, config.num_hidden_layers // 2, config.num_hidden_layers - 1]:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        # Check if the layer uses MoE by looking for a gate or router
        if hasattr(mlp, 'gate') or hasattr(mlp, 'router'):
            # Determine the number of experts in the model
            num_experts = None
            if hasattr(mlp, 'experts'):
                num_experts = len(mlp.experts)
            elif hasattr(mlp, 'expert_weights'):
                num_experts = len(mlp.expert_weights)
            
            # Check config for number of experts using possible field names
            config_num_experts = None
            for field in ['num_local_experts', 'n_routed_experts', 'num_experts']:
                config_num_experts = getattr(config, field, None)
                if config_num_experts is not None:
                    break
            assert config_num_experts is not None, f"Layer {layer_idx}: MoE detected but no num_experts field in config"
            if num_experts is not None:
                assert num_experts == config_num_experts, (
                    f"Layer {layer_idx}: Number of experts mismatch: config {config_num_experts}, model {num_experts}"
                )
            
            # Validate moe_intermediate_size if specified
            moe_intermediate_size = getattr(config, 'moe_intermediate_size', None)
            if moe_intermediate_size is not None and hasattr(mlp, 'experts'):
                for expert in mlp.experts:
                    assert expert.weight.shape[0] == moe_intermediate_size, (
                        f"Layer {layer_idx}: Expert weight shape mismatch: expected {moe_intermediate_size}, got {expert.weight.shape[0]}"
                    )
            print(f"Layer {layer_idx} MoE configuration validated ✓")
        else:
            print(f"Layer {layer_idx} is not MoE ✓")

def test_position_embeddings(model_setup):
    """Check max_position_embeddings and detect RoPE."""
    config = model_setup["config"]
    max_pos = getattr(config, "max_position_embeddings", 2048)  # Default if not specified
    assert max_pos > 0, f"max_position_embeddings must be positive, got {max_pos}"
    print(f"max_position_embeddings: {max_pos} ✓")
    if load_weights:
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
    config = model_setup["config"]
    num_layers = config.num_hidden_layers
    use_sliding_window = getattr(config, "use_sliding_window", False)
    if use_sliding_window:
        sliding_window = getattr(config, "sliding_window", None)
        assert sliding_window is not None, "sliding_window must be specified when use_sliding_window=true"
        assert isinstance(sliding_window, int) and sliding_window > 0, (
            f"sliding_window must be a positive integer, got {sliding_window}"
        )
        print(f"sliding_window: {sliding_window} ✓")
        max_window = getattr(config, "max_window_layers", num_layers)
        assert max_window <= num_layers, (
            f"max_window_layers ({max_window}) exceeds num_hidden_layers ({num_layers})"
        )
        print(f"max_window_layers ({max_window}) ≤ num_hidden_layers ({num_layers}) ✓")
    else:
        print("Sliding window disabled - max_window_layers and sliding_window are ignored")



if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])