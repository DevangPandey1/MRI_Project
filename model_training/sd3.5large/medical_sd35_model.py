# medical_sd35_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from typing import Dict, Optional, Tuple

class MedicalSD35MultiModal(nn.Module):
    """
    Multi-modal medical imaging model based on Stable Diffusion 3.5 Large
    Combines tabular clinical data with 3D MRI for improved prediction
    """
    def __init__(
        self,
        sd35_model_path: str = "stabilityai/stable-diffusion-3.5-large",
        num_clinical_features: int = 20,  # Adjust based on your ADNI features
        num_clinical_outcomes: int = 5,   # ADNI_MEM, ADNI_EF, etc.
        num_disease_classes: int = 3,     # CN, MCI, AD
        freeze_sd35_initially: bool = True,
        use_quantization: bool = False
    ):
        super().__init__()
        
        # Load pretrained SD3.5 components
        if use_quantization:
            self.transformer = self._load_quantized_transformer(sd35_model_path)
        else:
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                sd35_model_path,
                torch_dtype=torch.bfloat16
            )
            self.transformer = pipeline.transformer
            self.vae = pipeline.vae
            self.scheduler = pipeline.scheduler
        
        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size
        
        # Multi-modal encoders
        self.mri_encoder = self._create_mri_encoder()
        self.tabular_encoder = self._create_tabular_encoder(num_clinical_features)
        
        # Fusion module
        self.fusion_module = self._create_fusion_module()
        
        # Task-specific heads
        self.clinical_prediction_head = self._create_clinical_head(num_clinical_outcomes)
        self.disease_classification_head = self._create_classification_head(num_disease_classes)
        self.mri_reconstruction_head = self._create_reconstruction_head()
        
        # Optionally freeze SD3.5 weights initially
        if freeze_sd35_initially:
            self._freeze_sd35_components()
    
    def _load_quantized_transformer(self, model_path: str):
        """Load quantized version for lower VRAM usage"""
        from diffusers import BitsAndBytesConfig
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        transformer = SD3Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        return transformer
    
    def _create_mri_encoder(self):
        """
        3D MRI encoder using 3D convolutions
        Input: (B, 1, D, H, W) where D=144, H=192, W=144
        Output: (B, 512)
        """
        return nn.Sequential(
            # First conv block
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # /8
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # /16
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )
    
    def _create_tabular_encoder(self, num_features: int):
        """
        Tabular data encoder for clinical features
        Input: (B, num_features)
        Output: (B, 512)
        """
        return nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True)
        )
    
    def _create_fusion_module(self):
        """
        Cross-attention based fusion of MRI and tabular features
        """
        return nn.ModuleDict({
            'cross_attn': nn.MultiheadAttention(
                embed_dim=512,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            'feedforward': nn.Sequential(
                nn.Linear(512, 2048),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 512)
            ),
            'norm1': nn.LayerNorm(512),
            'norm2': nn.LayerNorm(512),
            'projection': nn.Linear(1024, self.hidden_size)
        })
    
    def _create_clinical_head(self, num_outcomes: int):
        """
        Head for predicting clinical outcomes (ADNI scores)
        """
        return nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_outcomes)
        )
    
    def _create_classification_head(self, num_classes: int):
        """
        Head for disease classification (CN, MCI, AD)
        """
        return nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
    
    def _create_reconstruction_head(self):
        """
        Head for MRI reconstruction using deconvolutions
        """
        return nn.Sequential(
            nn.Linear(self.hidden_size, 256 * 9 * 12 * 9),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 9, 12, 9)),
            
            # Deconv blocks
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # *2
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # *4
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # *8
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # *16
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def _freeze_sd35_components(self):
        """Freeze SD3.5 pretrained weights"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        if hasattr(self, 'vae'):
            for param in self.vae.parameters():
                param.requires_grad = False
    
    def unfreeze_sd35(self):
        """Unfreeze SD3.5 for fine-tuning"""
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        mri_data: torch.Tensor,
        tabular_data: torch.Tensor,
        task: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning
        
        Args:
            mri_data: (B, 1, D, H, W) 3D MRI scans
            tabular_data: (B, num_features) clinical features
            task: "clinical", "classification", "reconstruction", or "all"
        
        Returns:
            Dictionary with predictions for requested tasks
        """
        # Encode multi-modal inputs
        mri_features = self.mri_encoder(mri_data)  # (B, 512)
        tabular_features = self.tabular_encoder(tabular_data)  # (B, 512)
        
        # Cross-attention fusion
        # Reshape for attention: (B, 1, 512)
        mri_feat = mri_features.unsqueeze(1)
        tab_feat = tabular_features.unsqueeze(1)
        
        # Apply cross-attention
        attn_out, _ = self.fusion_module['cross_attn'](
            mri_feat, tab_feat, tab_feat
        )
        attn_out = self.fusion_module['norm1'](attn_out + mri_feat)
        
        # Feedforward
        ff_out = self.fusion_module['feedforward'](attn_out)
        fused_features = self.fusion_module['norm2'](ff_out + attn_out)
        fused_features = fused_features.squeeze(1)  # (B, 512)
        
        # Concatenate and project to transformer hidden size
        combined = torch.cat([mri_features, tabular_features], dim=1)  # (B, 1024)
        transformer_input = self.fusion_module['projection'](combined)  # (B, hidden_size)
        
        # Pass through SD3.5 transformer (for feature extraction)
        # Note: You may need to adapt this based on SD3.5's actual forward signature
        transformer_features = transformer_input
        
        # Task-specific predictions
        outputs = {}
        
        if task in ["clinical", "all"]:
            outputs['clinical_predictions'] = self.clinical_prediction_head(transformer_features)
        
        if task in ["classification", "all"]:
            outputs['disease_classification'] = self.disease_classification_head(transformer_features)
        
        if task in ["reconstruction", "all"]:
            outputs['mri_reconstruction'] = self.mri_reconstruction_head(transformer_features)
        
        return outputs
