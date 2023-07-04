import os
from typing import Any, Dict, List, Tuple

from vgproject.data.dataset import VGDataset
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.data_types import Split, BatchSample, BboxType
from vgproject.utils.misc import custom_collate, transform_sample
from vgproject.utils.config import Config
from vgproject.metrics.loss import Loss

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm
import optuna
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history


def objective(trial: Trial) -> float:
    cfg = Config.get_instance()  # type: ignore
    train_dataset: VGDataset = VGDataset(
        dir_path=cfg.dataset_path,
        split=Split.TRAIN,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
        preprocessed=True,
    )
    print("Train dataset created. Dataset length ", len(train_dataset))

    val_dataset: VGDataset = VGDataset(
        dir_path=cfg.dataset_path,
        split=Split.VAL,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
        preprocessed=True,
    )
    print("Validation dataset created. Dataset length: ", len(val_dataset))

    batch_size = trial.suggest_int(
        "batch_size",
        1,
        10,
    )
    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss is the weighted sum of the smooth l1 loss and the GIoU
    l = trial.suggest_float("l", 0.0, 1.0)
    loss_func = Loss(l)
    losses_list: List[float] = []
    accuracies_list: List[float] = []

    hidden_dim_1 = trial.suggest_int("hidden_dim_1", 512, 2048)
    hidden_dim_2 = trial.suggest_int("hidden_dim_2", 128, 512)
    if cfg.logging["resume"]:
        checkpoint: Dict[str, Any] = torch.load(cfg.logging["path"] + "model.pth")
        model = VGModel(hidden_dim_1, hidden_dim_2).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.model["gamma"]
        )
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch: int = checkpoint["epoch"]
        losses_list.append(checkpoint["loss"])
    else:
        model = VGModel(hidden_dim_1, hidden_dim_2).train()
        lr = trial.suggest_float("lr", 1e-5, 1e-2)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.model["gamma"]
        )
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs"):
        print("-------------------- Training --------------------------")
        epoch_loss = train_one_epoch(
            train_dataloader, model, loss_func, optimizer, device
        )
        losses_list.append(epoch_loss.cpu().item())
        lr_scheduler.step()

        # Evaluate on validation set for hyperparameter tuning
        print("-------------------- Validation ------------------------")
        accuracy = validate(val_dataloader, model, device)
        accuracies_list.append(accuracy)
        trial.report(accuracy, epoch)
        print(f"Accuracy: {accuracy} at epoch {epoch}")

        # Early stopping for non promising trials
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Save model after each epoch
        if cfg.logging["save_model"]:
            dir: str = cfg.logging["path"]
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
                f=f"{cfg.logging['path']}model.pth",
            )

        torch.clear_autocast_cache()

    return sum(accuracies_list) / len(accuracies_list)


def train_one_epoch(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    loss: Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tensor:
    # As loss we take smooth_l1 + GIoU
    epoch_loss_list: List[Tensor] = []

    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Move to gpu
        for sample in batch:
            sample.to(device)

        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        batch_loss: Tensor = loss.compute(out, bbox)
        epoch_loss_list.append(batch_loss)

        # Backward pass
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return torch.stack(epoch_loss_list).mean()


@torch.no_grad()
def validate(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    device: torch.device,
) -> float:
    # As accuracy we take the average IoU
    accuracy_list: List[Tensor] = []
    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Move to gpu
        for sample in batch:
            sample.to(device)

        # Forward pass
        out: Tensor = model(batch)

        accuracy_list.append(torch.diagonal(box_iou(out, bbox)).mean())

    return torch.stack(accuracy_list).mean().cpu().item()


def main() -> None:
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "train.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1, timeout=600)

    trial = study.best_trial
    print(f"Best hyperparameters: {trial.params}")
    fig = plot_optimization_history(study)
    fig.show()


if __name__ == "__main__":
    main()
