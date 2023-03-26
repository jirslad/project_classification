"""Vision Transformer from scratch
"""

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self,
                 input_channels:int=3,
                 patch_size:int=16,
                 embedding_dimension:int=768):
        super().__init__()
        self.patch_size = patch_size
        self.linear_projection = nn.Conv2d(in_channels=input_channels,
                                           out_channels=embedding_dimension,
                                           kernel_size=patch_size,
                                           stride=patch_size,
                                           padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # x of shape (batch, channels, height, width)
        assert x.shape[-1] % self.patch_size == 0, f"Image size ({x.shape[-1]}) must be multiple of patch size ({self.patch_size})"
        return self.flatten(self.linear_projection(x)).permute(0,2,1)


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 embedding_dimension:int=768,
                 heads:int=12,
                 dropout:float=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.multihead = nn.MultiheadAttention(embed_dim=embedding_dimension,
                                               num_heads=heads,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        self_attn, _ = self.multihead(x, x, x, need_weights=False)
        return self_attn


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 embedding_dimension:int=768,
                 mlp_units:int=3072,
                 dropout:float=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, mlp_units),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(mlp_units, embedding_dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dimension:int=768,
                 heads:int=12,
                 dropout_msa:float=0.0,
                 mlp_units:int=3072,
                 dropout_mlp:float=0.1) -> None:
        super().__init__()
        self.msa_block = MultiheadSelfAttention(embedding_dimension, heads, dropout_msa)
        self.mlp_block = MultiLayerPerceptron(embedding_dimension, mlp_units, dropout_mlp)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x
    

class ViT(nn.Module):
    ''' Vision Transformer model
    '''
    def __init__(self,
                 img_height:int=224,
                 img_width:int=224,
                 img_channels:int=3,
                 patch_size:int=16,
                 embedding_dimension:int=768,
                 encoder_layers:int=12,
                 msa_heads:int=12,
                 embedding_dropout:float=0.1,
                 msa_dropout:float=0.0,
                 mlp_dropout:float=0.1,
                 mlp_units:int=3072,
                 out_classes:int=1000):
        assert img_height % patch_size == 0, f"Image size ({img_height}) must be multiple of patch size ({patch_size})"
        assert img_width % patch_size == 0, f"Image size ({img_width}) must be multiple of patch size ({patch_size})"
        super().__init__()
        num_patches = int(img_height * img_width / patch_size**2)
        # embedding
        self.patch_embedding = PatchEmbedding(img_channels, patch_size, embedding_dimension)
        self.class_embedding = nn.Parameter(torch.randn((1, 1, embedding_dimension),
                                                        dtype=torch.float32,
                                                        requires_grad=True))
        self.position_embedding = nn.Parameter(torch.randn((1, num_patches+1, embedding_dimension),
                                                           dtype=torch.float32,
                                                           requires_grad=True))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        # transformer encoder
        # transformer_encoder_block = 
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dimension,
                                      msa_heads,
                                      msa_dropout,
                                      mlp_units,
                                      mlp_dropout)
            for _ in range(encoder_layers)])
        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, out_classes)
        )
        
    def forward(self, x):
        batch = x.shape[0]
        # embedding
        class_token = self.class_embedding.expand(batch, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:,0])
        return x
    


# ### DEBUGGING CODE
# img = torch.randn((3, 224, 224))
# img_batch = img.unsqueeze(dim=0)
# model = ViT(img_height=224,
#             img_width=224,
#             img_channels=3,
#             patch_size=16,
#             embedding_dimension=768,
#             encoder_layers=12,
#             msa_heads=12,
#             embedding_dropout=0.1,
#             msa_dropout=0.0,
#             mlp_dropout=0.1,
#             mlp_units=3072,
#             out_classes=3)
# from torchinfo import summary
# summary(model=model, 
#         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )
# logits = model(img_batch)
# print(logits)
# print(f"Input shape: {img_batch.shape}, output shape: {logits.shape}")