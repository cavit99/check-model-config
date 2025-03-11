# check_model_config/cli.py
import argparse
import sys
import pytest
import os

def main():
    """Run model configuration validation with the provided model path."""
    parser = argparse.ArgumentParser(
        description="Validate transformer model configurations."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model (Hugging Face repo or local directory). Example: OpenPipe/Deductive-Reasoning-Qwen-32B"
    )

    args = parser.parse_args()
    model_path = args.model

    # Set the model path as an environment variable for tests to access
    os.environ["CHECK_MODEL_PATH"] = model_path
    print("Running tests...please wait...")

    # Run pytest on the test file, relying on pyproject.toml for additional options
    exit_code = pytest.main(["check_model_config/tests.py"])
    sys.exit(exit_code)

if __name__ == "__main__":
    main()