"""
TAU NLP course 2025
Enhanced training loop with WandB and Hydra integration
"""

import os
import math
import logging
from typing import Optional, Dict, Any
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, StepLR
from torch.utils.data.dataloader import DataLoader
import wandb

from src.evaluator import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    """Enhanced trainer with WandB and Hydra integration"""

    def __init__(self, model, train_dataset, test_dataset, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
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

        # Initialize WandB if enabled
        self.use_wandb = cfg.tracker.mode != "disabled"
        if self.use_wandb and wandb.run is None:
            wandb.init(project=cfg.tracker.project, name=cfg.tracker.name, config=cfg)
            logger.info("WandB initialized for training")

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

    def save_checkpoint(self, results_dict):
        """Save model checkpoint"""
        if (
            hasattr(self.cfg.training, "ckpt_path")
            and self.cfg.training.ckpt_path is not None
        ):
            model_file_name = self.model.get_model_file_name()
            model_path = os.path.join(self.cfg.training.ckpt_path, model_file_name)
            # Save locally
            torch.save(self.model.state_dict(), model_path)
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
                description=f"Trained {model_file_name} model with seed {self.cfg.training.seed}",
                metadata=artifact_metadata,
            )

            # Add results to the run itself
            for key, value in artifact_metadata["metrics"].items():
                wandb.summary[key] = value

            artifact.add_file(model_path, name=model_file_name)

            # tags
            tags = [
                str(self.train_dataset.get_dataset_name()),
                "seed-" + str(self.cfg.training.seed),
            ]

            # Include the files under .hydra in the artifact
            # hydra_run_dir = HydraConfig.get().runtime.output_dir
            # results_file_name = "results" + model_file_name
            # if os.path.isdir(hydra_run_dir):
            #     artifact.add_file(os.path.join(hydra_run_dir, results_file_name), name=results_file_name)

            # Save the artifact to wandb
            wandb.log_artifact(artifact, tags=tags)

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
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                shuffle=self.cfg.data.get("shuffle", True),
            )

            losses = []
            all_predictions = []
            all_labels = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )

            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, labels=y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                # Collect predictions and labels for metrics calculation
                all_predictions.append(logits.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.training.grad_norm_clip
                    )
                    optimizer.step()

                    # Update learning rate
                    if scheduler is not None:
                        if step >= warmup_steps:
                            scheduler.step()
                        lr = optimizer.param_groups[0]["lr"]
                    else:
                        lr = self.cfg.training.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

                    # Log to WandB
                    if step % 10 == 0:  # Log every 10 steps
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
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Log epoch metrics
            avg_loss = np.mean(losses)
            if not is_train:
                logger.info("test loss: %f", avg_loss)
                evaluation_dict = self.evaluator.end_of_epoch_eval(
                    all_predictions, all_labels, prefix="eval"
                )
                res = {"eval/loss": avg_loss, "eval/epoch": epoch, **evaluation_dict}
                # TODO - maybe remove
                self.log_metrics(res)

                # Track best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # self.save_checkpoint(res)
            else:
                evaluation_dict = self.evaluator.end_of_epoch_eval(
                    all_predictions, all_labels, prefix="train"
                )
                res = {
                    "train/epoch_loss": avg_loss,
                    "train/epoch": epoch,
                    **evaluation_dict,
                }
                self.log_metrics(res)

            # Save checkpoint at end of each epoch
            # self.save_checkpoint(res)

        # Training loop
        for epoch in range(self.cfg.training.max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.cfg.training.max_epochs}")

            run_epoch("train")
            if self.test_dataset is not None:
                run_epoch("test")
        self.save_checkpoint({})

        # Log final metrics
        if self.use_wandb and wandb.run is not None:
            wandb.log({"final/best_loss": best_loss})
            wandb.finish()

        logger.info("Training completed!")
