# check_llm_config/cli.py
import argparse
import sys
import pytest

def main():
    """Run pytest with the provided model path."""
    parser = argparse.ArgumentParser(
        description="Validate transformer model configurations."
    )
    # Positional argument for model path (optional)
    parser.add_argument(
        "model_path", nargs="?", default=None,
        help="Path to the model (Hugging Face repo or local directory). Can be provided as a positional argument."
    )
    # Optional --model flag for backwards compatibility / user preference
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the model (Hugging Face repo or local directory)."
    )

    args = parser.parse_args()

    # Use --model if provided, otherwise fallback to positional model_path
    model_path = args.model or args.model_path

    if model_path is None:
        print("Error: Model path must be provided.")
        print("Usage: check-model-config <model_path> or check-model-config --model <model_path>")
        sys.exit(1)

    # Run pytest with the tests module and pass the model path.
    # Correctly group all pytest arguments into a single list.
    exit_code = pytest.main([
        "check_model_config/tests.py",
        "--model", model_path,
        "-v",
        "--tb=short"
    ])
    sys.exit(exit_code)

if __name__ == "__main__":
    main()