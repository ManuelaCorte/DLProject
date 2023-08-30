import gc
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from dotenv import load_dotenv
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.metrics.metrics import Metric, MetricsLogger
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split
from vgproject.utils.engines import train_one_epoch, validate
from vgproject.utils.misc import count_parameters, custom_collate, init_torch


def train(
    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    val_dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    device: torch.device,
    cfg: Config,
) -> Tuple[MetricsLogger, MetricsLogger]:
    train_metrics: MetricsLogger = MetricsLogger()
    val_metrics: MetricsLogger = MetricsLogger()

    # Loss is the weighted sum of the smooth l1 loss and the GIoU
    loss_func = Loss(cfg.train.l1, cfg.train.l2)

    model: VGModel = VGModel(cfg).train()
    # Separate parameters to train
    backbone_params: List[nn.Parameter] = [
        p for p in model.pretrained_model.parameters() if p.requires_grad
    ]

    # All parameters except the backbone parameters
    non_frozen_params = [
        p for p in set(model.parameters()) - set(model.pretrained_model.parameters())
    ]
    # print(len(backbone_params), len(non_frozen_params))
    optimizer = optim.AdamW(
        params=[
            {"params": backbone_params, "lr": cfg.train.lr_backbone, "weight_decay": 0},
            {"params": non_frozen_params, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay},
        ]
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.step_size)

    print("Model parameters: ", count_parameters(model))

    if cfg.logging.wandb:
        wandb.watch(model, loss_func, log="all", log_freq=100, log_graph=True)

    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        print("-------------------- Training --------------------------")
        epoch_train_metrics: Dict[str, float] = train_one_epoch(
            dataloader=train_dataloader,
            model=model,
            loss=loss_func,
            optimizer=optimizer,
            img_size=cfg.model.img_size,
            device=device,
        )

        lr_scheduler.step()

        train_metrics.update_metric(epoch_train_metrics)
        print("Training metrics at epoch ", epoch)
        print(epoch_train_metrics)

        # Evaluate on validation set for hyperparameter tuning
        print("-------------------- Validation ------------------------")
        epoch_val_metrics: Dict[str, float] = validate(
            dataloader=val_dataloader,
            model=model,
            loss=loss_func,
            img_size=cfg.model.img_size,
            device=device,
        )
        val_metrics.update_metric(epoch_val_metrics)
        print("Validation metrics at epoch ", epoch)
        print(epoch_val_metrics)

        if cfg.train.sweep:
            wandb.log(
                {
                    "validation_accuracy": epoch_val_metrics[Metric.IOU.value],
                }
            )
        # Log metrics to wandb putting train and val metrics together
        if cfg.logging.wandb:
            wandb.log(
                {
                    "Loss": {
                        "train": epoch_train_metrics[Metric.LOSS.value],
                        "val": epoch_val_metrics[Metric.LOSS.value],
                    },
                    "Average IOU": {
                        "train": epoch_train_metrics[Metric.IOU.value],
                        "val": epoch_val_metrics[Metric.IOU.value],
                    },
                    "Accuracy@25": {
                        "train": epoch_train_metrics[Metric.ACCURACY_25.value],
                        "val": epoch_val_metrics[Metric.ACCURACY_25.value],
                    },
                    "Accuracy@50": {
                        "train": epoch_train_metrics[Metric.ACCURACY_50.value],
                        "val": epoch_val_metrics[Metric.ACCURACY_50.value],
                    },
                    "Accuracy@75": {
                        "train": epoch_train_metrics[Metric.ACCURACY_75.value],
                        "val": epoch_val_metrics[Metric.ACCURACY_75.value],
                    },
                    "Accuracy@90": {
                        "train": epoch_train_metrics[Metric.ACCURACY_90.value],
                        "val": epoch_val_metrics[Metric.ACCURACY_90.value],
                    },
                },
                commit=True,
            )

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
                    "loss": epoch_train_metrics[Metric.LOSS.value],
                },
                f=f"{dir}model{epoch}.pth",
            )

        torch.cuda.empty_cache()
        gc.collect()

    return train_metrics, val_metrics


def initialize_run(sweep: bool = True) -> None:
    config = Config()
    wandb_run = None
    if sweep:
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb_run = wandb.init(project="vgproject")
        wandb_cfg = wandb.config
        config.update(wandb_cfg)
    else:
        if config.logging.wandb:
            load_dotenv()
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb_run = wandb.init(project="vgproject", config=config.as_dict())

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

    train_metrics, val_metrics = train(train_dataloader, val_dataloader, device, config)

    json.dump(train_metrics.metrics, open("../train_metrics.json", "w"))
    json.dump(val_metrics.metrics, open("../val_metrics.json", "w"))

    if config.logging.wandb:
        wandb_run.finish()  # type: ignore


def main() -> None:
    init_torch()
    cfg = Config()
    if cfg.train.sweep:
        sweep_configuration: Dict[str, Any] = json.load(
            open("../sweep_config.json", "r")
        )
        sweep: str = wandb.sweep(sweep_configuration, project="vgproject")
        wandb.agent(sweep_id='utlvbo45', function=initialize_run, count=10)
    else:
        initialize_run(cfg.train.sweep)


if __name__ == "__main__":
    main()
