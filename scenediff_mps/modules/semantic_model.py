"""
Semantic Model Module
=====================

Handles semantic feature extraction using DINOv3 and appearance-based cost computation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from .device import empty_device_cache

from utils import (
    predict_dinov2_hf_tokens,
    upsample_tokens,
)


class SemanticModel:
    """
    Handles semantic feature extraction and appearance-based cost computation
    using DINOv3.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize semantic model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.model = None
        self.feature_dim = config['models']['dinov3']['feature_dim']
        
    
    def _load_dinov3_model(self):
        """Load the semantic backbone.

        MPS port: DINOv3 weights are gated, so this loads DINOv2 from HF transformers
        (config models.dinov3.model_name, e.g. facebook/dinov2-large, feature_dim 1024).
        """
        from transformers import Dinov2Model

        model_name = self.config['models']['dinov3']['model_name']
        self.model = Dinov2Model.from_pretrained(model_name).to(self.device).eval()
        self._token_cache = {}
        print(f"[SemanticModel] Loaded DINOv2 model (DINOv3 substitute): {model_name}")

    def extract_features(self, image_np):
        """
        Extract DINO features from an image.

        MPS port: patch tokens are cached per image (the pipeline re-extracts the same frame
        once per paired view); the full-resolution map is re-upsampled on each call, so the
        output is identical to extracting from scratch.

        Args:
            image_np: Image as numpy array (H, W, 3)

        Returns:
            Feature map (C, H, W)
        """
        return upsample_tokens(self.extract_tokens(image_np), *image_np.shape[:2])

    def extract_tokens(self, image_np):
        """MPS port: cached (1, C, h, w) patch tokens; consumers upsample them in channel
        chunks (utils.feature_chunks) instead of holding the full-resolution map."""
        key = hash(image_np.tobytes())
        tokens = self._token_cache.get(key)
        if tokens is None:
            tokens = predict_dinov2_hf_tokens(image_np, self.model)
            self._token_cache[key] = tokens
        return tokens

    def extract_features_batch(self, images_np_list):
        """
        Extract features for a batch of images.
        
        Args:
            images_np_list: List of images as numpy arrays
            
        Returns:
            List of feature maps
        """
        features = []
        for img_np in images_np_list:
            feat = self.extract_features(img_np)
            features.append(feat)
        return features
    
    
    def compute_similarity(self, feat_1, feat_2):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            feat_1, feat_2: Feature vectors
            
        Returns:
            Similarity score
        """
        return torch.cosine_similarity(feat_1, feat_2, dim=0)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self._token_cache = {}
            empty_device_cache()
