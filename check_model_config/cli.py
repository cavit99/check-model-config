# check_model_config/cli.py
import argparse
import sys
import pytest
import os

def main():
    """Run model configuration validation with the provided model path and optional load_weights flag."""
    parser = argparse.ArgumentParser(
        description="Validate transformer model configurations with optional weight loading."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model (Hugging Face repo or local directory). Example: OpenPipe/Deductive-Reasoning-Qwen-32B"
    )
    parser.add_argument(
        "--load-weights",
        type=lambda x: x.lower() == "true",  # Convert string to bool
        default=True,
        help="Whether to load model weights (default: True). Set to False to skip weight-dependent tests."
    )

    args = parser.parse_args()
    model_path = args.model
    load_weights = args.load_weights

    # Set environment variables for tests to access
    os.environ["CHECK_MODEL_PATH"] = model_path
    os.environ["CHECK_LOAD_WEIGHTS"] = str(load_weights)
    print(f"Running tests with load_weights={load_weights}...please wait...")

    # Run pytest on the test file
    exit_code = pytest.main(["check_model_config/tests.py"])
    sys.exit(exit_code)

if __name__ == "__main__":
    main()