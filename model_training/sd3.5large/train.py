# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from typing import Dict, Optional

class MultiTaskTrainer:
    """
    Multi-task trainer for Medical SD3.5 model
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        task_weights: Optional[Dict[str, float]] = None,
        use_mixed_precision: bool = True,
        log_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Task loss weights
        self.task_weights = task_weights or {
            'clinical': 1.0,
            'classification': 1.0,
            'reconstruction': 0.5
        }
        
        # Optimizer with layer-wise learning rates
        self.optimizer = self._setup_optimizer(learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        
        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project="medical-sd35", name="multimodal-training")
    
    def _setup_optimizer(self, lr: float):
        """Setup optimizer with different learning rates for different components"""
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'transformer' in n and p.requires_grad],
                'lr': lr * 0.1  # Lower LR for pretrained transformer
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'transformer' not in n and p.requires_grad],
                'lr': lr  # Higher LR for new components
            }
        ]
        return optim.AdamW(param_groups, weight_decay=0.01)
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all task-specific losses"""
        losses = {}
        
        # Clinical prediction loss (MSE for regression)
        if 'clinical_predictions' in outputs:
            losses['clinical'] = self.mse_loss(
                outputs['clinical_predictions'],
                targets['clinical_outcomes']
            )
        
        # Disease classification loss (CrossEntropy)
        if 'disease_classification' in outputs:
            losses['classification'] = self.ce_loss(
                outputs['disease_classification'],
                targets['disease_labels']
            )
        
        # MRI reconstruction loss (L1 + MSE)
        if 'mri_reconstruction' in outputs:
            recon_mse = self.mse_loss(
                outputs['mri_reconstruction'],
                targets['mri_data']
            )
            recon_l1 = self.l1_loss(
                outputs['mri_reconstruction'],
                targets['mri_data']
            )
            losses['reconstruction'] = 0.5 * recon_mse + 0.5 * recon_l1
        
        # Weighted total loss
        total_loss = sum(
            self.task_weights.get(task, 1.0) * loss 
            for task, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'clinical': 0.0,
            'classification': 0.0,
            'reconstruction': 0.0,
            'total': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            mri_data = batch['mri'].to(self.device)
            tabular_data = batch['tabular'].to(self.device)
            clinical_outcomes = batch['clinical_outcomes'].to(self.device)
            disease_labels = batch['disease_labels'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                outputs = self.model(mri_data, tabular_data, task="all")
                
                targets = {
                    'clinical_outcomes': clinical_outcomes,
                    'disease_labels': disease_labels,
                    'mri_data': mri_data
                }
                
                losses = self.compute_losses(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'clinical': losses.get('clinical', torch.tensor(0.0)).item(),
                'classification': losses.get('classification', torch.tensor(0.0)).item()
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    f"train/{k}": v.item() for k, v in losses.items()
                })
        
        # Average losses
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {
            'clinical': 0.0,
            'classification': 0.0,
            'reconstruction': 0.0,
            'total': 0.0
        }
        
        correct_classifications = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            mri_data = batch['mri'].to(self.device)
            tabular_data = batch['tabular'].to(self.device)
            clinical_outcomes = batch['clinical_outcomes'].to(self.device)
            disease_labels = batch['disease_labels'].to(self.device)
            
            outputs = self.model(mri_data, tabular_data, task="all")
            
            targets = {
                'clinical_outcomes': clinical_outcomes,
                'disease_labels': disease_labels,
                'mri_data': mri_data
            }
            
            losses = self.compute_losses(outputs, targets)
            
            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            
            # Calculate classification accuracy
            if 'disease_classification' in outputs:
                preds = outputs['disease_classification'].argmax(dim=1)
                correct_classifications += (preds == disease_labels).sum().item()
                total_samples += disease_labels.size(0)
        
        # Average losses
        num_batches = len(self.val_loader)
        val_losses = {k: v / num_batches for k, v in val_losses.items()}
        
        # Add accuracy
        if total_samples > 0:
            val_losses['accuracy'] = correct_classifications / total_samples
        
        # Log to wandb
        if self.log_wandb:
            wandb.log({f"val/{k}": v for k, v in val_losses.items()})
        
        return val_losses
    
    def train(self, num_epochs: int, unfreeze_sd35_epoch: Optional[int] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            unfreeze_sd35_epoch: Epoch at which to unfreeze SD3.5 weights (optional)
        """
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Optionally unfreeze SD3.5 for fine-tuning
            if unfreeze_sd35_epoch and epoch == unfreeze_sd35_epoch:
                print("Unfreezing SD3.5 transformer for fine-tuning...")
                self.model.unfreeze_sd35()
                # Recreate optimizer with updated parameters
                self.optimizer = self._setup_optimizer(self.optimizer.param_groups[1]['lr'])
            
            # Train and validate
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nTrain Losses: {train_losses}")
            print(f"Val Losses: {val_losses}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses, is_best=True)
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_losses, is_best=False)
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")
