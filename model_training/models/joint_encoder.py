import torch
import torch.nn as nn
from .vit_MRI_encoder import ViT3DEncoder
from .tabular_bert_encoder import TabularBertEncoder

class CombinedEncoder(nn.Module):
    """
    A single encoder for 3D MRI and tabular data, adapted for JEPA and Latent Diffusion Model use.
    All core functionality is relevant to LDM with additional methods added to handle JEPA masking
    
    Changes for JEPA:
    1.  A learnable `mri_mask_token` is added.
    2.  The `forward` method now accepts an optional `mri_mask` tensor.
    3.  If a mask is provided, the embeddings of the corresponding MRI patches
        are replaced with the `mri_mask_token` before being processed by the
        ViT's Transformer encoder.
    """
    def __init__(
        self,
        # --- ViT Args ---
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        # --- Tabular Args ---
        num_continuous: int = 10,
        cat_cardinalities: list[int] = [],
        # --- Shared & Fusion Args ---
        embed_dim: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_fusion_layers: int = 3,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        # === Instantiate Encoders ===
        self.vit_encoder = ViT3DEncoder(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim, nhead=nhead, num_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.tab_encoder = TabularBertEncoder(num_continuous=num_continuous, cat_cardinalities=cat_cardinalities, embed_dim=embed_dim, nhead=nhead, num_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

        # === Fusion Module ===
        fusion_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.fusion_transformer = nn.TransformerDecoder(fusion_layer, num_layers=num_fusion_layers)

        # === JEPA Specific Token ===
        # Learnable token to replace masked MRI patch embeddings
        self.mri_mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x_mri: torch.Tensor, x_cont: torch.Tensor, x_cat: torch.Tensor, mri_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the combined encoder.

        Args:
            x_mri (torch.Tensor): MRI data.
            x_cont (torch.Tensor): Continuous tabular data.
            x_cat (torch.Tensor): Categorical tabular data.
            mri_mask (torch.Tensor, optional): A boolean mask for the MRI patches.
                Shape: (batch, num_patches). `True` indicates a patch to be masked.
                Defaults to None.

        Returns:
            torch.Tensor: The final fused conditioning embedding.
        """
        # 1. Get contextual embeddings for tabular data (always unmasked)
        tabular_embeddings = self.tab_encoder(x_cont, x_cat)

        # 2. Get MRI embeddings, applying the mask if provided for JEPA training
        mri_embeddings = self._encode_mri_with_masking(x_mri, mri_mask)

        # 3. Fuse using cross-attention
        fused_mri_embeddings = self.fusion_transformer(
            tgt=mri_embeddings,
            memory=tabular_embeddings
        )

        # 4. Concatenate for the final output
        final_conditioning = torch.cat([fused_mri_embeddings, tabular_embeddings], dim=1)
        return final_conditioning

    def _encode_mri_with_masking(self, x_mri: torch.Tensor, mri_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Helper function to encode MRI, handling the JEPA mask.
        """
        # This duplicates the logic from ViT3DEncoder's forward pass
        batch_size = x_mri.size(0)
        
        # Create patch embeddings
        # We need to access the sub-module of the imported ViTEncoder
        patches = self.vit_encoder.patch_embedding(x_mri)
        
        # Apply mask if provided (for JEPA context encoder)
        if mri_mask is not None:
            # Expand mask_token to match batch size and embed_dim
            mask_tokens = self.mri_mask_token.expand(batch_size, patches.shape[1], -1)
            # Use the mask to select where to place the mask tokens
            # Mask shape is (B, N_patches), we need (B, N_patches, 1) to broadcast
            mask = mri_mask.unsqueeze(-1).expand_as(patches)
            patches = torch.where(mask, mask_tokens, patches)

        # Prepend [CLS] token and add positional embedding
        cls_token_expanded = self.vit_encoder.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token_expanded, patches], dim=1)
        x += self.vit_encoder.pos_embedding

        # Pass through Transformer Encoder
        return self.vit_encoder.transformer_encoder(x)


# --- Usage Example ---
if __name__ == '__main__':
    # --- Shared Config ---
    BATCH_SIZE = 4
    EMBED_DIM = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_FUSION_LAYERS = 2
    DIM_FEEDFORWARD = 1024

    # --- MRI Config ---
    IMG_SIZE = (64, 64, 64)
    PATCH_SIZE = (16, 16, 16)
    IN_CHANNELS = 1
    NUM_MRI_PATCHES = (IMG_SIZE[0] // PATCH_SIZE[0])**3

    # --- Tabular Config ---
    NUM_CONTINUOUS = 5
    CAT_CARDINALITIES = [10, 3, 2] # 3 categorical features

    # --- Instantiate the Combined Encoder ---
    encoder = CombinedEncoder(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_continuous=NUM_CONTINUOUS,
        cat_cardinalities=CAT_CARDINALITIES,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_fusion_layers=NUM_FUSION_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )
    
    # --- Create Dummy Input Data ---
    dummy_mri = torch.randn(BATCH_SIZE, IN_CHANNELS, *IMG_SIZE)
    dummy_cont = torch.randn(BATCH_SIZE, NUM_CONTINUOUS)
    dummy_cat = torch.cat([torch.randint(0, card, (BATCH_SIZE, 1)) for card in CAT_CARDINALITIES], dim=1)

    # --- Create a Dummy Mask for JEPA Training ---
    # Mask 75% of the MRI patches
    mask_ratio = 0.75
    num_masked = int(mask_ratio * NUM_MRI_PATCHES)
    # Generate random indices to mask for each item in the batch
    masked_indices = torch.cat([torch.randperm(NUM_MRI_PATCHES)[:num_masked].unsqueeze(0) for _ in range(BATCH_SIZE)])
    
    # Create the boolean mask tensor
    jepa_mri_mask = torch.zeros(BATCH_SIZE, NUM_MRI_PATCHES, dtype=torch.bool)
    jepa_mri_mask.scatter_(1, masked_indices, True)

    print(f"Created a JEPA mask of shape: {jepa_mri_mask.shape}")
    print(f"Number of masked patches per sample: {jepa_mri_mask[0].sum().item()}")
    print("-" * 50)
    
    # --- Perform a Forward Pass with the Mask ---
    with torch.no_grad():
        # During JEPA training, you pass the mask to the context encoder
        context_embedding = encoder(dummy_mri, dummy_cont, dummy_cat, mri_mask=jepa_mri_mask)

    print(f"Output embedding shape: {context_embedding.shape}")