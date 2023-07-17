import gc
import json
import os
import pprint
from typing import Any, Dict, List, Tuple

import torch
from dotenv import load_dotenv
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

import wandb
from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split
from vgproject.utils.misc import custom_collate, init_torch


def train(
    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    val_dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    device: torch.device,
    cfg: Config,
) -> float:
    # Loss is the weighted sum of the smooth l1 loss and the GIoU
    loss_func = Loss(cfg.train.l1, cfg.train.l2)
    losses_list: List[float] = []
    accuracies_list: List[float] = []

    model = VGModel(cfg).train()

    # Separate parameters to train
    backbone_params: List[nn.Parameter] = [
        p for p in model.pretrained_model.parameters() if p.requires_grad
    ]

    # All parameters except the backbone
    non_frozen_params: List[nn.Parameter] = [
        p for p in model.fusion_module.parameters()
    ]
    non_frozen_params.extend(model.decoder.parameters())
    non_frozen_params.extend(model.reg_head.parameters())

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": cfg.train.lr_backbone},
            {"params": non_frozen_params, "lr": cfg.train.lr},
        ]
    )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=[cfg.train.lr_backbone, cfg.train.lr * 100],
        epochs=cfg.epochs,
        steps_per_epoch=len(train_dataloader),
    )

    if cfg.logging.wandb:
        wandb.watch(model, loss_func, log="all", log_freq=100, log_graph=True)

    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        print("-------------------- Training --------------------------")
        epoch_loss = train_one_epoch(
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            loss=loss_func,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            device=device,
            cfg=cfg,
        )
        losses_list.append(epoch_loss.item())
        lr_scheduler.step()

        # Evaluate on validation set for hyperparameter tuning
        print("-------------------- Validation ------------------------")
        accuracy = validate(val_dataloader, model, device)
        accuracies_list.append(accuracy)
        if cfg.logging.wandb:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train loss epoch": epoch_loss,
                    "Validation accuracy epoch": accuracy,
                },
                commit=True,
            )
        print(f"Accuracy: {accuracy} at epoch {epoch}")

        # Save model after each epoch
        if cfg.logging.save:
            dir: str = cfg.logging.path
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(
                obj={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": epoch_loss,
                },
                f=f"{dir}model{epoch}.pth",
            )

            if cfg.logging.wandb and not cfg.train.sweep:
                wandb.save(f"{dir}model{epoch}.pth")

        torch.cuda.empty_cache()
        gc.collect()

    if cfg.train.sweep:
        print("-------------------- Validation ------------------------")
        accuracy = validate(val_dataloader, model, device)
        accuracies_list.append(accuracy)
        if cfg.logging.wandb:
            wandb.log(
                {
                    "Validation accuracy": accuracy,
                },
                commit=True,
            )
    return sum(accuracies_list) / len(accuracies_list)


def train_one_epoch(
    epoch: int,
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    loss: Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    device: torch.device,
    cfg: Config,
) -> Tensor:
    # As loss we take smooth_l1 + GIoU
    epoch_loss_list: List[Tensor] = []

    for idx, (batch, bbox) in enumerate(tqdm(dataloader, desc="Batches")):
        optimizer.zero_grad()
        # Move to gpu
        for sample in batch:
            sample = sample.to(device)
        bbox = bbox.to(device)

        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        batch_loss: Tensor = loss.compute(out, bbox)

        # Backward pass
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss_list.append(batch_loss.detach())
        if (idx * len(batch)) % 4096 == 0:
            report: Dict[str, float] = {
                "Train loss": batch_loss.detach().item(),
                "Train accurracy": torch.diagonal(box_iou(out, bbox)).mean().item(),
            }
            if cfg.logging.wandb:
                wandb.log(report, commit=True)
            pprint.pprint(f"Batches: {idx}, {report}")

    return torch.stack(epoch_loss_list).mean()


@torch.no_grad()
def validate(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    device: torch.device,
) -> float:
    # As accuracy we take the average IoU
    model.eval()
    accuracy_list: List[Tensor] = []
    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Move to gpu
        for sample in batch:
            sample.to(device)
        bbox = bbox.to(device)

        # Forward pass
        out: Tensor = model(batch)

        accuracy_list.append(torch.diagonal(box_iou(out, bbox)).mean())

    return torch.stack(accuracy_list).mean().item()


def initialize_run(sweep: bool = True) -> None:
    config = Config()
    if sweep:
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="vgproject")
        wandb_cfg = wandb.config
        config.update(wandb_cfg)
    else:
        if config.logging.wandb:
            load_dotenv()
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(project="vgproject", config=config.as_dict())

    train_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TRAIN,
        output_bbox_type=BboxType.XYWH,
        augment=True,
        preprocessed=True,
    )
    print("Train dataset created. Dataset length ", len(train_dataset))

    val_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.VAL,
        output_bbox_type=BboxType.XYWH,
        augment=False,
        preprocessed=True,
    )
    print("Validation dataset created. Dataset length: ", len(val_dataset))

    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=custom_collate,
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(train_dataloader, val_dataloader, device, config)

    if config.logging.wandb:
        wandb.finish()


def main() -> None:
    init_torch()
    cfg = Config()
    if cfg.train.sweep:
        sweep_configuration: Dict[str, Any] = json.load(
            open("../sweep_config.json", "r")
        )
        sweep: str = wandb.sweep(sweep_configuration, project="vgproject")
        wandb.agent(sweep, function=initialize_run, count=10)
    else:
        initialize_run(cfg.train.sweep)


if __name__ == "__main__":
    main()
