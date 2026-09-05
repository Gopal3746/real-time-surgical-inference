import platform

import monai
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main() -> None:
    print("Surgical Video Inference")
    print("------------------------")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MONAI: {monai.__version__}")
    print(f"Device: {get_device()}")


if __name__ == "__main__":
    main()