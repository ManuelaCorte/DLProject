import os
from typing import Any, List, Tuple
from vgproject.data.dataset import VGDataset
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.bbox_types import BboxType
from vgproject.data.data_types import Split, BatchSample
from vgproject.utils.misc import custom_collate, transform_sample
from vgproject.utils.config import Config
from vgproject.utils.bbox_types import BboxType
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou_loss
from torch import Tensor
import torch
import torch.optim as optim
from tqdm import tqdm


def main() -> None:
    config = Config.get_instance()  # type: ignore
    dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
    )

    dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_epoch: List[float] = []

    if config.logging["resume"]:
        checkpoint: Any = torch.load(config.logging["path"] + "model.pth")
        model = VGModel().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.AdamW(model.parameters(), lr=config.model["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.model["gamma"]
        )
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch: int = checkpoint["epoch"]
        loss_epoch.append(checkpoint["loss"])

    model = VGModel().train()
    optimizer = optim.AdamW(model.parameters(), lr=config.model["lr"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.model["gamma"]
    )

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        loss: float = train_one_epoch(dataloader, model, optimizer).cpu().item()
        loss_epoch.append(loss)
        lr_scheduler.step()

        # Save model after each epoch
        if config.logging["save_model"]:
            dir = config.logging["path"]
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": loss,
                },
                f"{config.logging['path']}model.pth",
            )
        torch.clear_autocast_cache()
    print(loss_epoch)


def train_one_epoch(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    train_loss: List[Tensor] = []
    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        loss: Tensor = generalized_box_iou_loss(out, bbox, reduction="mean")
        train_loss.append(loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return torch.stack(train_loss).mean()


if __name__ == "__main__":
    main()
