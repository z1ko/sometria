
import torch as t
import torch.nn as nn
import lightning as L

class MotionWindowClassifier(L.LightningModule):
    def __init__(
        self,
        *,
        backbone,
        num_labels: int,
        freeze_backbone: bool = True,
        head: str = "linear",
        lr_head=1e-3,
        lr_backbone=1e-5
    ) -> None:

        super().__init__()