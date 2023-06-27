import os
from typing import List, Tuple
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
from clip import clip


def main() -> None:
    config = Config.get_instance()  # type: ignore
    dataset = VGDataset(
        dir_path=config.dataset["path"],
        split=Split.TRAIN,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
        transform_text=clip.tokenize,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.model["batch_size"],
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )

    model = VGModel(emb_dim=config.model["emb_dim"]).train()
    optimizer = optim.AdamW(model.parameters(), lr=config.model["lr"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.model["gamma"]
    )

    loss_epoch: List[Tensor] = []
    for epoch in tqdm(range(config.model["epochs"]), desc="Epochs"):
        loss_epoch.append(train_one_epoch(dataloader, model, optimizer))
        lr_scheduler.step()

        # Save model after each epoch
        if config.logging["save_model"]:
            dir = config.logging["path"]
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.state_dict(), f"{config.logging['path']}model_{epoch}.pth")

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
