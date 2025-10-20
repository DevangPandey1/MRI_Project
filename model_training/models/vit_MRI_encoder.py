import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class ViT3DEncoder(nn.Module):
    """
    3D Vision Transformer (ViT) Encoder for volumetric data like MRI scans.

    This model processes a 3D image by dividing it into a sequence of 3D patches,
    embedding them, and then passing them through a Transformer encoder. The output
    is a sequence of contextualized embeddings, suitable for conditioning a
    generative model like a latent diffusion model.

    Args:
        img_size (tuple[int, int, int]): The size of the input volume (Depth, Height, Width).
        patch_size (tuple[int, int, int]): The size of each 3D patch.
        in_channels (int): The number of input channels (e.g., 1 for grayscale MRI).
        embed_dim (int): The dimension of the embeddings.
        nhead (int): The number of heads in the multi-head attention models.
        num_layers (int): The number of sub-encoder-layers in the transformer encoder.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """
    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Calculate the number of patches in each dimension
        num_patches_d = img_size[0] // patch_size[0]
        num_patches_h = img_size[1] // patch_size[1]
        num_patches_w = img_size[2] // patch_size[2]
        num_patches = num_patches_d * num_patches_h * num_patches_w

        # --- Patching and Embedding ---
        # 1. We use a 3D convolution to create the patches. The kernel size and stride
        #    are both set to the patch size to create non-overlapping patches.
        #    The output channels are set to the embedding dimension.
        self.patch_embedding = nn.Sequential(
            nn.Conv3d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            # Rearrange the dimensions to be (batch, sequence, embedding_dim)
            Rearrange('b d n1 n2 n3 -> b (n1 n2 n3) d')
        )

        # 2. Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Learnable positional embeddings for each patch + [CLS] token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input 3D tensor (e.g., MRI scan).
                Shape: (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: The sequence of contextualized embeddings.
                Shape: (batch_size, num_patches + 1, embed_dim).
        """
        batch_size = x.size(0)

        # 1. Create patch embeddings
        # Shape: (batch_size, num_patches, embed_dim)
        patches = self.patch_embedding(x)
        
        # 2. Prepend [CLS] token
        # Shape: (batch_size, 1, embed_dim)
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        
        # 3. Concatenate [CLS] token and patch embeddings
        # Shape: (batch_size, num_patches + 1, embed_dim)
        x = torch.cat([cls_token_expanded, patches], dim=1)
        
        # 4. Add positional embeddings
        x += self.pos_embedding

        # 5. Pass through Transformer Encoder
        # Shape remains: (batch_size, num_patches + 1, embed_dim)
        transformer_output = self.transformer_encoder(x)
        
        return transformer_output

# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Configuration ---
    IMG_SIZE = (96, 96, 96)
    PATCH_SIZE = (16, 16, 16)
    IN_CHANNELS = 1
    EMBED_DIM = 768
    NUM_LAYERS = 6
    NHEAD = 8
    BATCH_SIZE = 4
    
    # Calculate expected sequence length for verification
    num_patches = (IMG_SIZE[0] // PATCH_SIZE[0]) * \
                  (IMG_SIZE[1] // PATCH_SIZE[1]) * \
                  (IMG_SIZE[2] // PATCH_SIZE[2])
    EXPECTED_SEQ_LENGTH = num_patches + 1

    # --- Instantiate the Encoder ---
    vit_encoder = ViT3DEncoder(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        nhead=NHEAD
    )

    print("ViT 3D Encoder Architecture:")
    # print(vit_encoder) # Can be very long, printing shapes is more informative
    print("-" * 50)
    
    # --- Create a Dummy 3D MRI Tensor ---
    # Shape: (batch_size, channels, depth, height, width)
    dummy_mri = torch.randn(BATCH_SIZE, IN_CHANNELS, *IMG_SIZE)
    print(f"Dummy MRI input shape: {dummy_mri.shape}")
    print("-" * 50)
    
    # --- Perform a Forward Pass ---
    with torch.no_grad():
        # This is the conditioning vector from the MRI
        mri_conditioning_embedding = vit_encoder(dummy_mri)

    print(f"Output embedding shape: {mri_conditioning_embedding.shape}")
    print(f"Expected shape: ({BATCH_SIZE}, {EXPECTED_SEQ_LENGTH}, {EMBED_DIM})")
    print("-" * 50)
    
    # --- Integration with Tabular Encoder ---
    print("Example of combining with TabularBertEncoder:")
    # Assume tabular_embedding is the output from your TabularBertEncoder
    # tabular_embedding shape: (BATCH_SIZE, NUM_TABULAR_TOKENS, EMBED_DIM)
    dummy_tabular_embedding = torch.randn(BATCH_SIZE, 7, EMBED_DIM) # From previous example
    
    # Concatenate along the sequence dimension (dim=1)
    final_conditioning = torch.cat([dummy_tabular_embedding, mri_conditioning_embedding], dim=1)
    
    print(f"Shape of tabular embedding: {dummy_tabular_embedding.shape}")
    print(f"Shape of MRI embedding:     {mri_conditioning_embedding.shape}")
    print(f"Shape of final combined conditioning vector: {final_conditioning.shape}")