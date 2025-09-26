"""
TAU NLP course 2025
Enhanced training loop with WandB and Hydra integration
"""

import os
import math
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

import wandb
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, StepLR
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from .consts import DatasetSplitName

from src.utils.evaluator import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    """Enhanced trainer with WandB and Hydra integration"""

    def __init__(self, model, train_dataset, eval_dataset, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cfg = cfg
        self.training_cfg = cfg.training
        self.evaluator = Evaluator()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS device for training")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for training")
        self.model = self.model.to(self.device)

        self.use_wandb = cfg.tracker.mode != "disabled"

    # # TODO: make it better with wandb artifacts
    # def save_checkpoint(self):
    #         ckpt_model = (
    #             self.model.module if hasattr(self.model, "module") else self.model
    #         )
    #         logger.info("saving %s", self.cfg.training.ckpt_path)
    #         torch.save(ckpt_model.state_dict(), self.cfg.training.ckpt_path)

    #         # Log checkpoint to WandB
    #         if self.use_wandb and wandb.run is not None:
    #             wandb.save(self.cfg.training.ckpt_path)

    @staticmethod
    def save_checkpoint(cfg, model, results_dict, tags=None):
        """Save model checkpoint"""
        if hasattr(cfg.training, "ckpt_path") and cfg.training.ckpt_path is not None:
            model_file_name = model.get_model_file_name()
            model_dir = Path(cfg.training.ckpt_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_file_name
            # Save locally
            torch.save(model.state_dict(), str(model_path))
            logger.info(f"Model state saved to {model_path}")

            # Prepare artifact for wandb
            art_name = f"{wandb.run.group}-{wandb.run.id}".replace("/", "-").replace(
                " ", ""
            )
            artifact_metadata = {
                "metrics": {**results_dict},
                "hydra_configs": HydraConfig.get().overrides.task,
            }
            artifact = wandb.Artifact(
                name=art_name,
                type="model",
                description=f"Trained {model_file_name} model with seed {cfg.training.seed}",
                metadata=artifact_metadata,
            )

            # Add results to the run itself
            for key, value in artifact_metadata["metrics"].items():
                wandb.summary[key] = value

            artifact.add_file(model_path, name=model_file_name)

            # Include the files under .hydra in the artifact
            # hydra_run_dir = HydraConfig.get().runtime.output_dir
            # results_file_name = "results" + model_file_name
            # if os.path.isdir(hydra_run_dir):
            #     artifact.add_file(os.path.join(hydra_run_dir, results_file_name), name=results_file_name)

            # Save the artifact to wandb
            wandb.log_artifact(artifact, tags=tags)

    def get_tags(self):
        return [
            str(self.train_dataset.get_dataset_name()),
            "seed-" + str(self.cfg.training.seed),
        ]

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB"""
        if self.use_wandb and wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def train(self):
        """Main training loop with WandB integration"""
        model = self.model

        # Log model architecture to WandB
        if self.use_wandb and wandb.run is not None:
            wandb.watch(model, log="all", log_freq=100)

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.cfg.training.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups,
            lr=self.cfg.training.learning_rate,
            betas=self.cfg.training.betas,
        )

        # Initialize learning rate scheduler
        total_steps = (
            len(self.train_dataset)
            // self.cfg.training.batch_size
            * self.cfg.training.max_epochs
        )
        if self.cfg.training.scheduler:
            warmup_steps = int(self.cfg.training.warmup_steps)
            if self.cfg.training.scheduler.scheduler_type == "StepLR":
                scheduler = StepLR(
                    optimizer,
                    step_size=self.cfg.training.scheduler.step_size,
                    gamma=self.cfg.training.scheduler.gamma,
                )
            elif self.cfg.training.scheduler.scheduler_type == "LinearLR":
                scheduler = LinearLR(
                    optimizer,
                    start_factor=self.cfg.training.scheduler.start_factor,
                    end_factor=self.cfg.training.scheduler.end_factor,
                    total_iters=total_steps,
                )
        else:
            warmup_steps = 0
            scheduler = None

        step = 0
        best_loss = float("inf")

        def run_epoch(split):
            nonlocal step, best_loss
            is_train = split == DatasetSplitName.TRAIN
            model.train(is_train)
            data = self.train_dataset if is_train else self.eval_dataset
            loader = DataLoader(
                data,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                shuffle=self.cfg.data.get("shuffle", True),
            )

            losses = []
            all_predictions = []
            all_labels = []
            mlm_logits = []
            mlm_labels = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )

            for it, batch in pbar:
                # Detect MLM batch by dict keys

                if self.cfg.training.is_mlm:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                else:
                    # Classification batch: (x, y)
                    input_ids = batch[0].to(self.device)
                    attention_mask = None
                    labels = batch[1].to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = loss.mean()
                    losses.append(loss.item())
                if not self.cfg.training.is_mlm:
                    probs = torch.softmax(logits, dim=-1)
                    all_predictions.append(probs.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())
                else:
                    # Collect logits and labels for MLM evaluation
                    mlm_logits.append(logits.detach().cpu().numpy())
                    mlm_labels.append(labels.detach().cpu().numpy())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.training.grad_norm_clip
                    )
                    optimizer.step()

                    if scheduler is not None:
                        if step >= warmup_steps:
                            scheduler.step()
                        lr = optimizer.param_groups[0]["lr"]
                    else:
                        lr = self.cfg.training.learning_rate

                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

                    if step % 10 == 0:
                        self.log_metrics(
                            {
                                "train/loss": loss.item(),
                                "train/learning_rate": lr,
                                "train/step": step,
                            },
                            step,
                        )

                step += 1

            # Calculate metrics
            if not self.cfg.training.is_mlm:
                all_predictions = np.concatenate(all_predictions, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

            # Log epoch metrics
            avg_loss = np.mean(losses)
            if not is_train:
                logger.info("%s loss: %f", split, avg_loss)
                if not self.cfg.training.is_mlm:
                    evaluation_dict = self.evaluator.end_of_epoch_eval(
                        all_predictions, all_labels, prefix=split
                    )
                    res = {
                        "eval/loss": avg_loss,
                        "eval/epoch": epoch,
                        **evaluation_dict,
                    }
                else:
                    # MLM: evaluate accuracy
                    evaluation_dict = self.evaluator.mlm_eval(
                        mlm_logits, mlm_labels, prefix=split
                    )
                    res = {
                        "eval/loss": avg_loss,
                        "eval/epoch": epoch,
                        **evaluation_dict,
                    }
                if avg_loss < best_loss:
                    best_loss = avg_loss
            else:
                if not self.cfg.training.is_mlm:
                    evaluation_dict = self.evaluator.end_of_epoch_eval(
                        all_predictions, all_labels, prefix=split
                    )
                    res = {
                        "train/epoch_loss": avg_loss,
                        "train/epoch": epoch,
                        **evaluation_dict,
                    }
                else:
                    # MLM: evaluate accuracy
                    evaluation_dict = self.evaluator.mlm_eval(
                        mlm_logits, mlm_labels, prefix=split
                    )
                    res = {
                        "train/epoch_loss": avg_loss,
                        "train/epoch": epoch,
                        **evaluation_dict,
                    }
            self.log_metrics(res)

        if self.cfg.training.start_with_eval and self.eval_dataset is not None:
            epoch = -1
            run_epoch(DatasetSplitName.VALIDATION)
        # Training loop
        for epoch in range(self.cfg.training.max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.cfg.training.max_epochs}")

            run_epoch(DatasetSplitName.TRAIN)
            if self.eval_dataset is not None:
                run_epoch(DatasetSplitName.VALIDATION)

        self.save_checkpoint(self.cfg, self.model, {}, self.get_tags())

        # Log final metrics
        if self.use_wandb and wandb.run is not None:
            wandb.log({"final/best_loss": best_loss})
            wandb.finish()

        logger.info("Training completed!")
