import torch
from monai.networks.nets import EfficientNetBN

from surgphase.annotations import PHASES

MODEL_NAME = "efficientnet-b0"
NUM_PHASES = len(PHASES)


class SurgicalPhaseClassifier(torch.nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.network = EfficientNetBN(
            model_name=MODEL_NAME,
            pretrained=pretrained,
            progress=pretrained,
            spatial_dims=2,
            in_channels=3,
            num_classes=NUM_PHASES,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)