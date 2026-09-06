import platform

import monai
import torch

from surgphase.device import get_device


def main() -> None:
    device = get_device()

    print("Surgical Video Inference")
    print("------------------------")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MONAI: {monai.__version__}")
    print(f"Device: {device.type}")


if __name__ == "__main__":
    main()