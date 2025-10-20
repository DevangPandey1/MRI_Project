import torch
import torch.nn as nn
import math

class TabularBertEncoder(nn.Module):
    """
    Tabular BERT Encoder for creating conditioning vectors for generative models.

    This model is adapted from the TabularBert architecture to serve as an
    encoder. Instead of predicting a final value, it outputs a sequence of
    contextualized embeddings that can be used as a conditioning signal for
    other models, such as a latent diffusion model.

    The key changes from the original `TabularBert` are:
    1.  The prediction head (`self.head`) is removed.
    2.  The `forward` method returns the full output sequence from the
        Transformer encoder. This sequence includes the [CLS] token embedding
        and the embeddings for all continuous and categorical features.

    This output sequence can be concatenated with embeddings from other encoders
    (e.g., a ViT for images) and fed into the cross-attention layers of a
    diffusion model's U-Net.

    Args:
        num_continuous (int): Number of continuous features.
        cat_cardinalities (list[int]): A list containing the number of unique
            categories for each categorical feature.
        embed_dim (int): The embedding dimension for all features.
        nhead (int): The number of heads in the multi-head attention models.
        num_layers (int): The number of sub-encoder-layers in the transformer encoder.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list[int],
        embed_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        num_categorical = len(cat_cardinalities)
        total_features = num_categorical + num_continuous + 1  # +1 for [CLS] token

        # --- Learnable Embeddings ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities]
        )
        self.cont_norm = nn.LayerNorm(num_continuous)
        self.cont_projection = nn.Linear(num_continuous, embed_dim)
        self.column_embeddings = nn.Embedding(total_features, embed_dim)

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

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x_cont (torch.Tensor): Tensor of continuous features.
                Shape: (batch_size, num_continuous).
            x_cat (torch.Tensor): Tensor of categorical features.
                Shape: (batch_size, num_categorical).

        Returns:
            torch.Tensor: The sequence of contextualized embeddings.
                Shape: (batch_size, total_features, embed_dim).
        """
        batch_size = x_cont.size(0)

        # Process continuous and categorical features into embeddings
        cont_embed = self.cont_projection(self.cont_norm(x_cont)).unsqueeze(1)
        cat_embeds = [
            embed(x_cat[:, i]).unsqueeze(1) for i, embed in enumerate(self.cat_embeddings)
        ]
        cat_embed = torch.cat(cat_embeds, dim=1)

        # Prepend [CLS] token and concatenate all features
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token_expanded, cont_embed, cat_embed], dim=1)

        # Add column embeddings for positional context
        column_indices = torch.arange(x.size(1), device=x.device)
        x += self.column_embeddings(column_indices)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x)

        # Return the full sequence of embeddings
        return transformer_output

# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Configuration ---
    NUM_CONTINUOUS = 10
    CAT_CARDINALITIES = [12, 6, 2, 24, 4]
    EMBED_DIM = 64
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 256
    BATCH_SIZE = 32
    
    # Total number of "tokens" or features the model will output
    # 1 (CLS) + 1 (for the block of continuous features) + 5 (categorical features) = 7
    EXPECTED_SEQ_LENGTH = 1 + 1 + len(CAT_CARDINALITIES)


    # --- Instantiate the Encoder ---
    encoder = TabularBertEncoder(
        num_continuous=NUM_CONTINUOUS,
        cat_cardinalities=CAT_CARDINALITIES,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )

    print("Encoder Architecture:")
    print(encoder)
    print("-" * 50)

    # --- Create Dummy Input Data ---
    x_continuous_sample = torch.randn(BATCH_SIZE, NUM_CONTINUOUS)
    x_categorical_sample = torch.cat(
        [torch.randint(0, card, (BATCH_SIZE, 1)) for card in CAT_CARDINALITIES],
        dim=1
    )

    # --- Perform a Forward Pass ---
    with torch.no_grad():
        # This is the conditioning vector for your diffusion model
        conditioning_embedding = encoder(x_continuous_sample, x_categorical_sample)

    print(f"Output embedding shape: {conditioning_embedding.shape}")
    print(f"Expected shape: ({BATCH_SIZE}, {EXPECTED_SEQ_LENGTH}, {EMBED_DIM})")
    
    # You would then concatenate this with your ViT's output embeddings
    # For example:
    # vit_embeddings = vit_encoder(mri_images) # Shape: (BATCH_SIZE, NUM_VIT_PATCHES, EMBED_DIM)
    # final_conditioning = torch.cat([conditioning_embedding, vit_embeddings], dim=1)
    # And feed `final_conditioning` into your diffusion model's U-Net.