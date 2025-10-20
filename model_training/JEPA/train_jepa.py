"""
Contrastive JEPA training script for multi-modal (3D MRI + tabular) data.

This script wires together the combined encoder and predictor defined in
model_training/models to train a C-JEPA model with a VICReg objective.

This pre-trained encoder will then be used as the encoder for a Latent Diffusion model
The latent diffusion model will be responsible for decoding the MRI and tabular data seperately.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = lambda x, *args, **kwargs: x  # type: ignore

try:
    from model_training.dataloader import MRITabularDataset
    from model_training.models.joint_encoder import CombinedEncoder
    from model_training.models.predictor import PredictorModel
except ImportError:  # pragma: no cover - fallback when run as a module
    CURRENT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = CURRENT_DIR.parent
    if str(REPO_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(REPO_ROOT))
    from dataloader import MRITabularDataset  # type: ignore
    from models.joint_encoder import CombinedEncoder  # type: ignore
    from models.predictor import PredictorModel  # type: ignore


# ---------------------------------------------------------------------------
# Tuneable hyper-parameters
# ---------------------------------------------------------------------------
SEED: int = 1337
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Data
TRAIN_BATCH_SIZE: int = 4
VAL_BATCH_SIZE: int = 4
NUM_WORKERS: int = 4

# JEPA / masking
MRI_IMG_SIZE: Tuple[int, int, int] = (144, 192, 144)
MRI_PATCH_SIZE: Tuple[int, int, int] = (16, 16, 16)
MRI_IN_CHANNELS: int = 1
MRI_MASK_RATIO: float = 0.8  # fraction of MRI patches masked for context encoder

# Encoders
EMBED_DIM: int = 512
ENCODER_NUM_HEADS: int = 8
ENCODER_LAYERS: int = 8
FUSION_LAYERS: int = 3
ENCODER_DIM_FEEDFORWARD: int = 2048
ENCODER_DROPOUT: float = 0.1

# Predictor
PREDICTOR_EMBED_DIM: int = 256
PREDICTOR_DEPTH: int = 6
PREDICTOR_HEADS: int = 8
PREDICTOR_MLP_RATIO: float = 4.0
PREDICTOR_DROPOUT: float = 0.0
PREDICTOR_ATTN_DROPOUT: float = 0.0

# Optimisation
LEARNING_RATE: float = 0.005
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 100
GRAD_CLIP_NORM: float | None = 1.0
USE_AMP: bool = True
EMA_DECAY: float = 0.996  # target encoder momentum

# VICReg weights
VICREG_INVARIANCE_WEIGHT: float = 25.0
VICREG_VARIANCE_WEIGHT: float = 25.0
VICREG_COVARIANCE_WEIGHT: float = 1.0

# Logging / checkpoints
LOG_INTERVAL: int = 20
VAL_INTERVAL: int = 1
CHECKPOINT_INTERVAL: int = 5
OUTPUT_DIR: Path = Path("model_training/JEPA/checkpoints")

NUM_MRI_PATCHES: int = (
    (MRI_IMG_SIZE[0] // MRI_PATCH_SIZE[0])
    * (MRI_IMG_SIZE[1] // MRI_PATCH_SIZE[1])
    * (MRI_IMG_SIZE[2] // MRI_PATCH_SIZE[2])
)

"""
run = wandb.init(
    entity="ALPACA-3D",
    project="JEPA Encoder Training",
    config = {
        "batch_size": TRAIN_BATCH_SIZE,
        "mri_img_size": MRI_IMG_SIZE,
        "mri_patch_size": MRI_PATCH_SIZE,
        "mri_in_channels": MRI_IN_CHANNELS,
        "mri_mask_ratio": MRI_MASK_RATIO,
        "embed_dim": EMBED_DIM,
        "encoder_num_heads": ENCODER_NUM_HEADS,
        "encoder_layers": ENCODER_LAYERS,
        "fusion_layers": FUSION_LAYERS,
        "encoder_dim_feedforward": ENCODER_DIM_FEEDFORWARD,
        "encoder_dropout": ENCODER_DROPOUT,
        "predictor_embed_dim": PREDICTOR_EMBED_DIM,
        "predictor_depth": PREDICTOR_DEPTH,
        "predictor_heads": PREDICTOR_HEADS,
        "predictor_mlp_ratio": PREDICTOR_MLP_RATIO,
        "predictor_dropout": PREDICTOR_DROPOUT,
        "predictor_attn_dropout": PREDICTOR_ATTN_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "ema_decay": EMA_DECAY,
        "vicreg_invariance_weight": VICREG_INVARIANCE_WEIGHT,
        "vicreg_variance_weight": VICREG_VARIANCE_WEIGHT,
        "vicreg_covariance_weight": VICREG_COVARIANCE_WEIGHT,
        "log_interval": LOG_INTERVAL,
        "val_interval": VAL_INTERVAL,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "num_mri_patches": NUM_MRI_PATCHES
    }

)
"""

@dataclass(frozen=True)
class FeatureSplit:
    continuous: List[str]
    categorical: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_schema(schema_path: Path) -> Dict:
    with schema_path.open("r") as f:
        return json.load(f)


def resolve_feature_split(
    df: pd.DataFrame,
    schema: Dict,
    extra_continuous: Sequence[str] | None = None,
) -> FeatureSplit:
    """Determine which tabular columns are treated as continuous vs categorical."""
    meta_cols = {"subject_id", "IMAGEUID"}
    available_cols = [c for c in df.columns if c not in meta_cols]

    base_cont = list(schema.get("cont_obs_cols", []))
    extras = list(extra_continuous or [])
    continuous = [c for c in base_cont + extras if c in available_cols]
    continuous = list(dict.fromkeys(continuous))  # preserve order

    model_cols = [c for c in schema.get("model_input_cols", []) if c in available_cols]
    if not model_cols:
        model_cols = [c for c in available_cols if c != "mri_path"]

    categorical = [c for c in model_cols if c not in continuous]
    categorical = list(dict.fromkeys(categorical))

    return FeatureSplit(continuous=continuous, categorical=categorical)


def prepare_dataframe(csv_path: Path, feature_split: FeatureSplit) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = ["mri_path"] + feature_split.continuous + feature_split.categorical

    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df = df[expected_cols].copy()
    df = df.dropna(subset=["mri_path"])
    df = df[df["mri_path"].astype(str).str.strip() != ""]
    return df


def build_dataloader(
    df: pd.DataFrame,
    feature_split: FeatureSplit,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Tuple[DataLoader, List[int], List[int]]:
    dataset = MRITabularDataset(df)
    _ = dataset.tabular_cols  # preserves ordering within the dataset

    num_cont = len(feature_split.continuous)
    cont_indices = list(range(num_cont))
    cat_indices = list(range(num_cont, num_cont + len(feature_split.categorical)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )
    return loader, cont_indices, cat_indices


def sample_mri_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample boolean masks and indices for masked/unmasked MRI patches."""
    num_masked = max(1, int(mask_ratio * num_patches))
    all_indices = torch.arange(num_patches, device=device)

    masked_indices = torch.stack(
        [torch.randperm(num_patches, device=device)[:num_masked].sort().values for _ in range(batch_size)],
        dim=0,
    )

    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, masked_indices, True)

    visible_indices = torch.stack([all_indices[~mask[i]] for i in range(batch_size)], dim=0)
    return mask, masked_indices, visible_indices


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather token embeddings at specified indices."""
    expand_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=expand_indices)


def vicreg_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    sim_weight: float,
    var_weight: float,
    cov_weight: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute VICReg loss between predicted and target embeddings."""
    batch, num_tokens, dim = predicted.shape
    predicted = predicted.view(batch * num_tokens, dim)
    target = target.view(batch * num_tokens, dim)

    invariance = F.mse_loss(predicted, target)

    def variance_term(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x = x - x.mean(dim=0)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def covariance_term(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.size(0) - 1)
        cov = cov - torch.diagonal(cov).diag()
        return cov.pow(2).sum() / dim

    variance = variance_term(predicted)
    covariance = covariance_term(predicted)

    loss = sim_weight * invariance + var_weight * variance + cov_weight * covariance
    return loss


def update_momentum_encoder(
    online_model: nn.Module,
    target_model: nn.Module,
    momentum: float,
) -> None:
    with torch.no_grad():
        for online_param, target_param in zip(online_model.parameters(), target_model.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    epoch: int,
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    feature_split: FeatureSplit,
    path: Path,
) -> None:
    ensure_output_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "online_encoder": online_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "feature_split": feature_split.__dict__,
        },
        path,
    )


def forward_pass(
    encoder: CombinedEncoder,
    batch: Dict[str, torch.Tensor],
    mri_mask: torch.Tensor | None,
    cont_indices: Sequence[int],
    cat_indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run encoder forward pass and return separated MRI/tabular tokens."""
    mri = batch["mri"].to(device, non_blocking=True)
    tabular = batch["tabular"].to(device, non_blocking=True)

    x_cont = tabular[:, cont_indices]
    x_cat = tabular[:, cat_indices] if cat_indices else torch.zeros(tabular.size(0), 0, device=device)

    if x_cat.numel() > 0:
        x_cat = torch.round(x_cat).clamp(min=0, max=1).long()
    else:
        x_cat = x_cat.to(torch.long)

    outputs = encoder(mri, x_cont, x_cat, mri_mask=mri_mask)

    mri_token_count = NUM_MRI_PATCHES + 1  # +1 for [CLS]

    mri_tokens = outputs[:, :mri_token_count]
    tab_tokens = outputs[:, mri_token_count:]
    return mri_tokens, tab_tokens


def train_one_epoch(
    epoch: int,
    online_encoder: CombinedEncoder,
    target_encoder: CombinedEncoder,
    predictor: PredictorModel,
    train_loader: DataLoader,
    cont_indices: Sequence[int],
    cat_indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    online_encoder.train()
    predictor.train()

    running_loss = 0.0
    num_batches = 0
    num_patches = NUM_MRI_PATCHES

    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch:03d}")  # type: ignore
    use_postfix = hasattr(progress, "set_postfix")
    for step, batch in progress:
        # print("MRI: ", batch["mri"].shape)
        # print("Tabular: ",batch["tabular"].shape)
        
        batch_size = batch["mri"].shape[0]
        mri_mask, masked_indices, visible_indices = sample_mri_masks(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=MRI_MASK_RATIO,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            # NOTE: Not using the tabular tokens here. We are doing cross-attention in the encoder but
            # Need to make sure that performance isn't better when using both
            context_tokens, _ = forward_pass(
                online_encoder,
                batch,
                mri_mask=mri_mask,
                cont_indices=cont_indices,
                cat_indices=cat_indices,
                device=device,
            )

            with torch.no_grad():
                target_tokens, _ = forward_pass(
                    target_encoder,
                    batch,
                    mri_mask=None,
                    cont_indices=cont_indices,
                    cat_indices=cat_indices,
                    device=device,
                )

            context_patch_tokens = context_tokens[:, 1:]  # drop CLS
            target_patch_tokens = target_tokens[:, 1:]

            context_visible = gather_tokens(context_patch_tokens, visible_indices)
            target_masked = gather_tokens(target_patch_tokens, masked_indices)

            # TODO: Ensure that the predictor is correct (concerned about the target_mask feature since we shouldn't be masking anything)
            predictions = predictor(
                context_visible,
                target_masked.detach(),
                masks_ctxt=visible_indices,
                masks_tgt=masked_indices,
            )

            loss = vicreg_loss(
                predictions,
                target_masked.detach(),
                sim_weight=VICREG_INVARIANCE_WEIGHT,
                var_weight=VICREG_VARIANCE_WEIGHT,
                cov_weight=VICREG_COVARIANCE_WEIGHT,
            )

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(online_encoder.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(online_encoder.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        update_momentum_encoder(online_encoder, target_encoder, EMA_DECAY)

        running_loss += loss.item()
        num_batches += 1
        """
        if step % LOG_INTERVAL == 0:
            if use_postfix:
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            else:
                print(f"[Epoch {epoch:03d} | step {step:04d}] loss={loss.item():.4f}")
        """

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def evaluate(
    online_encoder: CombinedEncoder,
    target_encoder: CombinedEncoder,
    predictor: PredictorModel,
    data_loader: DataLoader,
    cont_indices: Sequence[int],
    cat_indices: Sequence[int],
    device: torch.device,
) -> float:
    online_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    total_loss = 0.0
    num_batches = 0
    num_patches = NUM_MRI_PATCHES

    for batch in data_loader:
        batch_size = batch["mri"].shape[0]
        mri_mask, masked_indices, visible_indices = sample_mri_masks(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=MRI_MASK_RATIO,
            device=device,
        )

        context_tokens, _ = forward_pass(
            online_encoder,
            batch,
            mri_mask=mri_mask,
            cont_indices=cont_indices,
            cat_indices=cat_indices,
            device=device,
        )

        target_tokens, _ = forward_pass(
            target_encoder,
            batch,
            mri_mask=None,
            cont_indices=cont_indices,
            cat_indices=cat_indices,
            device=device,
        )

        context_patch_tokens = context_tokens[:, 1:]
        target_patch_tokens = target_tokens[:, 1:]

        context_visible = gather_tokens(context_patch_tokens, visible_indices)
        target_masked = gather_tokens(target_patch_tokens, masked_indices)

        predictions = predictor(
            context_visible,
            target_masked,
            masks_ctxt=visible_indices,
            masks_tgt=masked_indices,
        )

        loss = vicreg_loss(
            predictions,
            target_masked,
            sim_weight=VICREG_INVARIANCE_WEIGHT,
            var_weight=VICREG_VARIANCE_WEIGHT,
            cov_weight=VICREG_COVARIANCE_WEIGHT,
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def create_models(
    num_continuous: int,
    num_categorical: int,
    device: torch.device,
) -> Tuple[CombinedEncoder, CombinedEncoder, PredictorModel]:
    cat_cardinalities = [2] * num_categorical

    online_encoder = CombinedEncoder(
        img_size=MRI_IMG_SIZE,
        patch_size=MRI_PATCH_SIZE,
        in_channels=MRI_IN_CHANNELS,
        num_continuous=num_continuous,
        cat_cardinalities=cat_cardinalities,
        embed_dim=EMBED_DIM,
        nhead=ENCODER_NUM_HEADS,
        num_encoder_layers=ENCODER_LAYERS,
        num_fusion_layers=FUSION_LAYERS,
        dim_feedforward=ENCODER_DIM_FEEDFORWARD,
        dropout=ENCODER_DROPOUT,
    ).to(device)

    target_encoder = copy.deepcopy(online_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad_(False)

    predictor = PredictorModel(
        img_size=144,
        patch_size=16,
        num_frames=12,
        tubelet_size=1,
        embed_dim=EMBED_DIM,
        predictor_embed_dim=PREDICTOR_EMBED_DIM,
        depth=PREDICTOR_DEPTH,
        num_heads=PREDICTOR_HEADS,
        mlp_ratio=PREDICTOR_MLP_RATIO,
        drop_rate=PREDICTOR_DROPOUT,
        attn_drop_rate=PREDICTOR_ATTN_DROPOUT,
        use_mask_tokens=False,
    ).to(device)

    return online_encoder, target_encoder, predictor


def build_optimizer(
    online_encoder: CombinedEncoder,
    predictor: PredictorModel,
) -> AdamW:
    params = list(online_encoder.parameters()) + list(predictor.parameters())
    optimizer = AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer


def main() -> None:
    set_seed(SEED)

    device = torch.device(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    base_dir = Path(__file__).resolve().parents[1]
    train_csv = base_dir / "X_train.csv"
    val_csv = base_dir / "X_val.csv"
    schema_path = base_dir / "columns_schema.json"

    schema = load_schema(schema_path)
    raw_df = pd.read_csv(train_csv, nrows=100)  # small probe for split inference
    feature_split = resolve_feature_split(
        raw_df,
        schema=schema,
        extra_continuous=["PTEDUCAT", "APOE4", "next_visit_months", "months_since_bl"],
    )

    train_df = prepare_dataframe(train_csv, feature_split)
    val_loader = None
    if val_csv.exists():
        val_df = prepare_dataframe(val_csv, feature_split)
        val_loader, _, _ = build_dataloader(
            val_df,
            feature_split=feature_split,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    else:
        print(f"[Warning] Validation file not found at {val_csv}. Skipping validation.")

    train_loader, cont_indices, cat_indices = build_dataloader(
        train_df,
        feature_split=feature_split,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    online_encoder, target_encoder, predictor = create_models(
        num_continuous=len(feature_split.continuous),
        num_categorical=len(feature_split.categorical),
        device=device,
    )

    optimizer = build_optimizer(online_encoder, predictor)

    ensure_output_dir(OUTPUT_DIR)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            epoch,
            online_encoder,
            target_encoder,
            predictor,
            train_loader,
            cont_indices,
            cat_indices,
            optimizer,
            scaler,
            device,
        )

        if val_loader is not None and epoch % VAL_INTERVAL == 0:
            val_loss = evaluate(
                online_encoder,
                target_encoder,
                predictor,
                val_loader,
                cont_indices,
                cat_indices,
                device,
            )
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            #run.log({"Training loss": train_loss, "Validation Loss": val_loss})
        else:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}")
            # Log to Weights and Biases
            #run.log({"Training loss": train_loss})

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == NUM_EPOCHS:
            ckpt_path = OUTPUT_DIR / f"cjepa_epoch{epoch:03d}.pt"
            save_checkpoint(
                epoch,
                online_encoder,
                target_encoder,
                predictor,
                optimizer,
                scaler,
                feature_split,
                ckpt_path,
            )


if __name__ == "__main__":
    main()
