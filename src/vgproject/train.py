import os
from typing import Any, Dict, List, Tuple

from vgproject.data.dataset import VGDataset
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.bbox_types import BboxType
from vgproject.data.data_types import Split, BatchSample
from vgproject.utils.misc import custom_collate, transform_sample
from vgproject.utils.config import Config
from vgproject.utils.bbox_types import BboxType
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
    config = Config.get_instance()  # type: ignore
    train_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
    )

    val_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
    )

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
    loss = Loss(l)
    losses: List[float] = []
    accuracies: List[float] = []

    hidden_dim_1 = trial.suggest_int("hidden_dim_1", 512, 2048)
    hidden_dim_2 = trial.suggest_int("hidden_dim_2", 128, 512)
    if config.logging["resume"]:
        checkpoint: Dict[str, Any] = torch.load(config.logging["path"] + "model.pth")
        model = VGModel(hidden_dim_1, hidden_dim_2).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.model["gamma"]
        )
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch: int = checkpoint["epoch"]
        losses.append(checkpoint["loss"])
    else:
        model = VGModel(hidden_dim_1, hidden_dim_2).train()
        lr = trial.suggest_float("lr", 1e-5, 1e-2)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.model["gamma"]
        )
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epochs"):
        print("-------------------- Training --------------------------")
        loss_epoch = train_one_epoch(train_dataloader, model, loss, optimizer)
        losses.append(loss_epoch.cpu().item())
        lr_scheduler.step()

        # Evaluate on validation set for hyperparameter tuning
        print("-------------------- Validation ------------------------")
        accuracy = validate(val_dataloader, model)
        accuracies.append(accuracy)
        trial.report(accuracy, epoch)
        print(f"Accuracy: {accuracy} at epoch {epoch}")

        # Early stopping for non promising trials
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Save model after each epoch
        if config.logging["save_model"]:
            dir: str = config.logging["path"]
            if not os.path.exists(dir):
                os.makedirs(dir)

            torch.save(
                obj={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": loss_epoch,
                },
                f=f"{config.logging['path']}model.pth",
            )

        torch.clear_autocast_cache()

    return sum(accuracies) / len(accuracies)


def train_one_epoch(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    loss: Loss,
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    # As loss we take smooth_l1 + GIoU
    train_loss: List[Tensor] = []

    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        batch_loss: Tensor = loss.compute(out, bbox)
        train_loss.append(batch_loss)

        # Backward pass
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return torch.stack(train_loss).mean()


@torch.no_grad()
def validate(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]], model: VGModel
) -> float:
    # As accuracy we take the average IoU
    accuracy: List[Tensor] = []
    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Forward pass
        out: Tensor = model(batch)

        accuracy.append(torch.diagonal(box_iou(out, bbox)).mean())

    return torch.stack(accuracy).mean().cpu().item()


def main() -> None:
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "test"
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
