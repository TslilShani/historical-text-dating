"""
TAU NLP course 2025
Enhanced training loop with WandB and Hydra integration
"""

import math
import logging
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader
import wandb

logger = logging.getLogger(__name__)


class Trainer:
    """Enhanced trainer with WandB and Hydra integration"""

    def __init__(self, model, train_dataset, test_dataset, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfg = cfg

        # Set device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Initialize WandB if enabled
        use_wandb = cfg.tracker.mode != "disabled"
        if use_wandb and wandb.run is None:
            wandb.init(project=cfg.tracker.project, name=cfg.tracker.name, config=cfg)
            logger.info("WandB initialized for training")

    def save_checkpoint(self):
        """Save model checkpoint"""
        if hasattr(self.cfg, "ckpt_path") and self.cfg.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info("saving %s", self.cfg.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.cfg.ckpt_path)

            # Log checkpoint to WandB
            if self.cfg.tracker.mode != "disabled" and wandb.run is not None:
                wandb.save(self.cfg.ckpt_path)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB"""
        if self.cfg.tracker.mode != "disabled" and wandb.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def train(self):
        """Main training loop with WandB integration"""
        model = self.model

        # Log model architecture to WandB
        if self.cfg.tracker.mode != "disabled" and wandb.run is not None:
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
            {"params": params_decay, "weight_decay": self.cfg.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=self.cfg.learning_rate, betas=self.cfg.betas
        )

        # Initialize learning rate scheduler
        total_steps = (
            len(self.train_dataset) // self.cfg.batch_size * self.cfg.max_epochs
        )
        if self.cfg.lr_decay:
            scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=self.cfg.warmup_tokens
            )
        else:
            scheduler = None

        step = 0
        best_loss = float("inf")

        def run_epoch(split):
            nonlocal step, best_loss
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers
            )

            losses = []
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
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.grad_norm_clip
                    )
                    optimizer.step()

                    # Update learning rate
                    if scheduler is not None:
                        scheduler.step()

                    # decay the learning rate based on our progress
                    if self.cfg.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step
                        if self.tokens < self.cfg.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, self.cfg.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - self.cfg.warmup_tokens
                            ) / float(
                                max(1, self.cfg.final_tokens - self.cfg.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = self.cfg.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = self.cfg.learning_rate

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

            # Log epoch metrics
            avg_loss = np.mean(losses)
            if not is_train:
                logger.info("test loss: %f", avg_loss)
                self.log_metrics({"eval/loss": avg_loss, "eval/epoch": epoch})

                # Track best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint()
            else:
                self.log_metrics({"train/epoch_loss": avg_loss, "train/epoch": epoch})

        # Initialize token counter for learning rate decay
        self.tokens = 0

        # Training loop
        for epoch in range(self.cfg.max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.cfg.max_epochs}")

            run_epoch("train")
            if self.test_dataset is not None:
                run_epoch("test")

            # Save checkpoint at end of each epoch
            self.save_checkpoint()

        # Log final metrics
        if self.cfg.use_wandb and wandb.run is not None:
            wandb.log({"final/best_loss": best_loss})
            wandb.finish()

        logger.info("Training completed!")
