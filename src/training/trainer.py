from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.ec2vae       import EC2VAE, ForwardOutput
from src.models.grl          import grad_reverse
from src.training.loss       import compute_loss, LossOutput
from src.training.scheduler  import BetaScheduler


class Trainer:
    def __init__(
        self,
        model:        EC2VAE,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       dict,
        device:       torch.device,
        project_root: str,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device
        self.ckpt_dir     = Path(project_root) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        t = config["training"]
        self.total_epochs = t["epochs"]           
        self.gamma        = t["gamma"]           
        self.pos_weight   = t.get("pos_weight", 5.0)
        self.swap_recon_ratio = t.get("swap_recon_ratio", 0.3)
        self.style_cls_weight = float(t.get("style_cls_weight", 0.0))
        self.transfer_style_weight = float(t.get("transfer_style_weight", 0.0))
        self.adv_style_weight = float(t.get("adv_style_weight", 0.0))
        self.content_consistency_weight = float(t.get("content_consistency_weight", 0.0))
        self.orth_weight = float(t.get("orth_weight", 0.0))
        self.grl_lambda = float(t.get("grl_lambda", 1.0))

        self.early_stopping_patience = int(t.get("early_stopping_patience", 0))
        self.early_stopping_min_delta = float(t.get("early_stopping_min_delta", 0.0))
        self.early_stopping_metric = t.get("early_stopping_metric", "val/recon")

        z_r_dim = int(config["model"]["z_r_dim"])
        z_p_dim = int(config["model"]["z_p_dim"])
        n_classes = int(config.get("model", {}).get("n_styles", 2))
        self.style_head = nn.Linear(z_r_dim, n_classes).to(self.device)
        self.style_adv_head = nn.Linear(z_p_dim, n_classes).to(self.device)

        self.optimizer  = Adam(
            list(model.parameters())
            + list(self.style_head.parameters())
            + list(self.style_adv_head.parameters()),
            lr=t["lr"],
        )
        self.scheduler  = BetaScheduler.from_config(config)
        self.use_amp    = bool(t.get("use_amp", device.type == "cuda")) and (device.type == "cuda")
        self.scaler     = GradScaler(enabled=self.use_amp)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.swap_warmup_epochs: int = t.get("swap_warmup_epochs", 20)

        self.start_epoch:    int   = 0
        self.best_metric_value: float = float("inf")
        self.best_epoch:     int   = -1
        self.no_improve_epochs: int = 0
        self.history: list[dict]   = []   


    def load_latest(self) -> bool:
        ckpt_path = self.ckpt_dir / "latest.pt"
        if not ckpt_path.exists():
            print("No checkpoint found — training from scratch.")
            return False

        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])

        if "style_head_state" in ckpt:
            self.style_head.load_state_dict(ckpt["style_head_state"])
        if "style_adv_head_state" in ckpt:
            self.style_adv_head.load_state_dict(ckpt["style_adv_head_state"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError:
            print("Optimizer state is incompatible with current model; reinitializing optimizer.")


        self.start_epoch   = ckpt["epoch"] + 1
        self.best_metric_value = ckpt.get(
            "best_metric_value",
            ckpt.get("best_val_loss", float("inf")),
        )
        self.best_epoch = ckpt.get("best_epoch", -1)
        self.no_improve_epochs = ckpt.get("no_improve_epochs", 0)
        self.history       = ckpt.get("history", [])
        print(
            f"Resumed from epoch {ckpt['epoch']}  "
            f"(best {self.early_stopping_metric}={self.best_metric_value:.4f})"
        )
        return True

    def _save_checkpoints(self, epoch: int, val_metrics: dict[str, float]) -> None:
        metric_val = self._monitor_value(val_metrics)
        val_total = val_metrics["val/total"]

        torch.save(
            {
                "epoch":          epoch,
                "model_state":    self.model.state_dict(),
                "style_head_state": self.style_head.state_dict(),
                "style_adv_head_state": self.style_adv_head.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_metric_value": self.best_metric_value,
                "best_val_loss":  self.best_metric_value,
                "early_stopping_metric": self.early_stopping_metric,
                "best_epoch":     self.best_epoch,
                "no_improve_epochs": self.no_improve_epochs,
                "history":        self.history,
            },
            str(self.ckpt_dir / "latest.pt"),
        )

        if metric_val < self.best_metric_value:
            self.best_metric_value = metric_val
            self.best_epoch = epoch
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "style_head_state": self.style_head.state_dict(),
                    "style_adv_head_state": self.style_adv_head.state_dict(),
                    "val_loss":    val_total,
                    "monitor_metric": self.early_stopping_metric,
                    "monitor_value": metric_val,
                },
                str(self.ckpt_dir / "best_model.pt"),
            )
            print(
                f" New best model saved  "
                f"({self.early_stopping_metric}={metric_val:.4f}, "
                f"val/total={val_total:.4f})"
            )

    def _monitor_value(self, val_metrics: dict[str, float]) -> float:
        if self.early_stopping_metric not in val_metrics:
            available = ", ".join(sorted(val_metrics.keys()))
            raise KeyError(
                f"Unknown early_stopping_metric '{self.early_stopping_metric}'. "
                f"Available metrics: {available}"
            )
        return float(val_metrics[self.early_stopping_metric])

    def _early_stopping_update(
        self,
        epoch: int,
        current_metric_value: float,
        significant_improved: bool,
    ) -> bool:
        if self.early_stopping_patience <= 0:
            return False

        if significant_improved:
            self.no_improve_epochs = 0
            return False

        self.no_improve_epochs += 1
        remaining = self.early_stopping_patience - self.no_improve_epochs
        print(
            f"  EarlyStopping: no improvement for {self.no_improve_epochs}/"
            f"{self.early_stopping_patience} epochs "
            f"({self.early_stopping_metric}: "
            f"best={self.best_metric_value:.4f} @ epoch {self.best_epoch}, "
            f"current={current_metric_value:.4f}, "
            f"min_delta={self.early_stopping_min_delta:.6f})"
        )
        if remaining <= 0:
            print(f"  EarlyStopping triggered at epoch {epoch}.")
            return True
        return False

    @staticmethod
    def _to_fp32_output(output: ForwardOutput) -> ForwardOutput:
        return ForwardOutput(
            logits=output.logits.float(),
            rhythm_pred=output.rhythm_pred.float(),
            mu_p=output.mu_p.float(),
            logvar_p=output.logvar_p.float(),
            mu_r=output.mu_r.float(),
            logvar_r=output.logvar_r.float(),
            swap_logits=output.swap_logits.float() if output.swap_logits is not None else None,
            swap_rhythm_pred=(
                output.swap_rhythm_pred.float()
                if output.swap_rhythm_pred is not None else None
            ),
            swap_perm=output.swap_perm,
        )

    @staticmethod
    def _print_key_metrics(epoch: int, total_epochs: int, beta: float, elapsed: float,
                           train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        print(
            f"Epoch {epoch:>3}/{total_epochs - 1}  β={beta:.4f}  ({elapsed:.0f}s)"
        )
        print(
            "  train: "
            f"total={train_metrics['loss/total']:.4f}  "
            f"recon={train_metrics['loss/recon']:.4f}  "
            f"kl={train_metrics['loss/kl']:.4f}  "
            f"rhythm={train_metrics['loss/rhythm']:.4f}  "
            f"swap_recon={train_metrics['loss/swap_recon']:.4f}  "
            f"swap_rhythm={train_metrics['loss/swap_rhythm']:.4f}  "
            f"style_cls={train_metrics.get('loss/style_cls', 0.0):.4f}  "
            f"style_adv={train_metrics.get('loss/style_adv', 0.0):.4f}  "
            f"content_cons={train_metrics.get('loss/content_consistency', 0.0):.4f}  "
            f"orth={train_metrics.get('loss/orth', 0.0):.4f}"
        )
        print(
            "  val:   "
            f"total={val_metrics['val/total']:.4f}  "
            f"recon={val_metrics['val/recon']:.4f}  "
            f"kl={val_metrics['val/kl']:.4f}  "
            f"rhythm={val_metrics['val/rhythm']:.4f}  "
            f"style_cls={val_metrics.get('val/style_cls', 0.0):.4f}  "
            f"style_adv={val_metrics.get('val/style_adv', 0.0):.4f}  "
            f"orth={val_metrics.get('val/orth', 0.0):.4f}"
        )

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        beta = self.scheduler.get(epoch)

        if self.swap_warmup_epochs > 0:
            swap_weight = min(1.0, epoch / self.swap_warmup_epochs)
        else:
            swap_weight = 1.0

        totals: dict[str, float] = {
            "loss/total": 0.0,
            "loss/recon": 0.0,
            "loss/kl":    0.0,
            "loss/rhythm": 0.0,
            "loss/swap_recon":  0.0,
            "loss/swap_rhythm": 0.0,
            "loss/style_cls": 0.0,
            "loss/transfer_style_cls": 0.0,
            "loss/style_adv": 0.0,
            "loss/content_consistency": 0.0,
            "loss/orth": 0.0,
        }
        n_batches = 0

        for x, rhythm_gt, label in self.train_loader:
            x         = x.to(self.device)
            rhythm_gt = rhythm_gt.to(self.device)
            label     = label.to(self.device).long()
            x_f       = x.float()
            rhythm_f  = rhythm_gt.float()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                output = self.model(x, do_swap=True)
            output = self._to_fp32_output(output)

            swap_rhythm_gt = None
            if output.swap_perm is not None:
                swap_rhythm_gt = rhythm_f[output.swap_perm]

            loss_out = compute_loss(
                output, x_f, rhythm_f,
                beta=beta,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                swap_weight=swap_weight,
                swap_recon_ratio=self.swap_recon_ratio,
                swap_rhythm_gt=swap_rhythm_gt,
            )

            style_logits = self.style_head(output.mu_r)
            style_cls_loss = F.cross_entropy(style_logits, label)
            adv_logits = self.style_adv_head(grad_reverse(output.mu_p, self.grl_lambda))
            style_adv_loss = F.cross_entropy(adv_logits, label)

            content_consistency_loss = torch.tensor(0.0, device=self.device)
            if output.swap_logits is not None:
                transferred_probs = torch.sigmoid(output.swap_logits)
                _, _, mu_p_t, _, _, _ = self.model.encode(transferred_probs)
                content_consistency_loss = F.smooth_l1_loss(mu_p_t, output.mu_p.detach())

            z_p_c = output.mu_p - output.mu_p.mean(dim=0, keepdim=True)
            z_r_c = output.mu_r - output.mu_r.mean(dim=0, keepdim=True)
            cross_cov = (z_p_c.T @ z_r_c) / max(z_p_c.size(0), 1)
            orth_loss = (cross_cov.pow(2)).mean()

            transfer_style_loss = torch.tensor(0.0, device=self.device)
            if (
                self.transfer_style_weight > 0.0
                and output.swap_logits is not None
                and output.swap_perm is not None
            ):
                transferred_probs = torch.sigmoid(output.swap_logits)
                _, _, _, _, mu_r_t, _ = self.model.encode(transferred_probs)
                transfer_style_logits = self.style_head(mu_r_t)
                transfer_targets = label[output.swap_perm]
                transfer_style_loss = F.cross_entropy(transfer_style_logits, transfer_targets)

            total_loss = (
                loss_out.total
                + self.style_cls_weight * style_cls_loss
                + self.transfer_style_weight * transfer_style_loss
                + self.adv_style_weight * style_adv_loss
                + self.content_consistency_weight * content_consistency_loss
                + self.orth_weight * orth_loss
            )

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                nn.utils.clip_grad_norm_(self.style_head.parameters(), max_norm=3.0)
                nn.utils.clip_grad_norm_(self.style_adv_head.parameters(), max_norm=3.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                nn.utils.clip_grad_norm_(self.style_head.parameters(), max_norm=3.0)
                nn.utils.clip_grad_norm_(self.style_adv_head.parameters(), max_norm=3.0)
                self.optimizer.step()

            step_stats = loss_out.as_dict()
            step_stats["loss/total"] = total_loss.item()
            step_stats["loss/style_cls"] = style_cls_loss.item()
            step_stats["loss/transfer_style_cls"] = transfer_style_loss.item()
            step_stats["loss/style_adv"] = style_adv_loss.item()
            step_stats["loss/content_consistency"] = content_consistency_loss.item()
            step_stats["loss/orth"] = orth_loss.item()
            for k, v in step_stats.items():
                totals[k] += v
            n_batches += 1

        epoch_metrics = {k: v / n_batches for k, v in totals.items()}
        epoch_metrics["swap_weight"] = swap_weight
        epoch_metrics["swap_recon_ratio"] = self.swap_recon_ratio
        return epoch_metrics

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        beta = self.scheduler.get(epoch)

        totals: dict[str, float] = {
            "val/total":  0.0,
            "val/recon":  0.0,
            "val/kl":     0.0,
            "val/rhythm": 0.0,
            "val/style_cls": 0.0,
            "val/style_adv": 0.0,
            "val/orth": 0.0,
        }
        n_batches = 0

        with torch.no_grad():
            for x, rhythm_gt, label in self.val_loader:
                x         = x.to(self.device)
                rhythm_gt = rhythm_gt.to(self.device)
                label     = label.to(self.device).long()
                x_f       = x.float()
                rhythm_f  = rhythm_gt.float()

                with autocast(enabled=self.use_amp):
                    # do_swap=False: skip style-swap branch for cleaner val metric
                    output   = self.model(x, do_swap=False)
                output = self._to_fp32_output(output)

                loss_out = compute_loss(
                    output, x_f, rhythm_f,
                    beta=beta,
                    gamma=self.gamma,
                    pos_weight=self.pos_weight,
                )

                style_logits = self.style_head(output.mu_r)
                style_cls_loss = F.cross_entropy(style_logits, label)
                adv_logits = self.style_adv_head(output.mu_p)
                style_adv_loss = F.cross_entropy(adv_logits, label)

                z_p_c = output.mu_p - output.mu_p.mean(dim=0, keepdim=True)
                z_r_c = output.mu_r - output.mu_r.mean(dim=0, keepdim=True)
                cross_cov = (z_p_c.T @ z_r_c) / max(z_p_c.size(0), 1)
                orth_loss = (cross_cov.pow(2)).mean()

                val_total = (
                    loss_out.total
                    + self.style_cls_weight * style_cls_loss
                    + self.adv_style_weight * style_adv_loss
                    + self.orth_weight * orth_loss
                )

                totals["val/total"]  += val_total.item()
                totals["val/recon"]  += loss_out.recon.item()
                totals["val/kl"]     += loss_out.kl.item()
                totals["val/rhythm"] += loss_out.rhythm.item()
                totals["val/style_cls"] += style_cls_loss.item()
                totals["val/style_adv"] += style_adv_loss.item()
                totals["val/orth"] += orth_loss.item()
                n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    def train(self) -> list[dict]:
        print(
            f"Training EC2-VAE  |  "
            f"epochs {self.start_epoch}–{self.total_epochs - 1}  |  "
            f"device={self.device}  |  amp={self.use_amp}"
        )
        if self.early_stopping_patience > 0:
            print(
                f"Early stopping enabled  |  patience={self.early_stopping_patience}  "
                f"metric={self.early_stopping_metric}  "
                f"min_delta={self.early_stopping_min_delta}"
            )

        for epoch in range(self.start_epoch, self.total_epochs):
            t0   = time.time()
            beta = self.scheduler.get(epoch)

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            elapsed = time.time() - t0
            monitor_value = self._monitor_value(val_metrics)

            epoch_log = {
                "epoch": epoch,
                "beta":  beta,
                "time":  elapsed,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_log)

            self._print_key_metrics(
                epoch=epoch,
                total_epochs=self.total_epochs,
                beta=beta,
                elapsed=elapsed,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            prev_best_metric = self.best_metric_value
            self._save_checkpoints(epoch, val_metrics)

            significant_improved = (prev_best_metric - monitor_value) > self.early_stopping_min_delta
            if self._early_stopping_update(epoch, monitor_value, significant_improved):
                break

        print("Training complete.")
        return self.history