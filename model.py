import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange




class image_embedding(nn.Module) :
  def __init__(self, in_channel: int, img_size: int, patch_size: int, emb_dim: int) :
    super().__init__()

    self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size, p2=patch_size)
    self.linear = nn.Linear(patch_size * patch_size * in_channel, emb_dim)

    self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
    
    n_patches = img_size * img_size // patch_size**2
    self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

  def forward(self, x) :
    # x: [batch, channel, width, height]
    batch, _, _, _ = x.shape

    # patch flatten
    x = self.rearrange(x)  # [bach_size, patch_num, patch_size * patch_size * in_channel]

    # patch embedding
    x = self.linear(x)  # [bach_size, patch_num, emb_dim]

    c = repeat(self.cls_token, '() n d -> b n d', b=batch)  # [bach_size, patch_num, emb_dim]
    x = torch.cat((c, x), dim=1)  # [bach_size, patch_num + 1, emb_dim]
    x = x + self.positions  # [bach_size, patch_num + 1, emb_dim]

    return x


class MultiHeadAttention(nn.Module) :
    def __init__(self, emb_dim: int, num_heads: int, dropout_ratio: float) :
        super().__init__()

        self.emb_dim = emb_dim 
        self.num_heads = num_heads 
        self.scaling = (self.emb_dim // num_heads) ** -0.5
        
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)
                
    def forward(self, x) :
        # query: [batch_size, patch_num + 1, emb_dim]
        # key: [batch_size, patch_num + 1, emb_dim]
        # value: [batch_size, patch_num + 1, emb_dim]

        Q = self.query(x)  # [batch_size, patch_num + 1, emb_dim]
        K = self.key(x)  # [batch_size, patch_num + 1, emb_dim]
        V = self.value(x)  # [batch_size, patch_num + 1, emb_dim]

        Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
        K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)  # For Q * K(Transpose)
        V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)

        # scaled dot-product
        attention = torch.matmul(Q, K)  # [b, h, q, k]
        attention = attention * self.scaling  # [b, h, q, k]
        
        attention = torch.softmax(attention, dim=-1)  # [b, h, q, k]
        attention = self.dropout(attention)  # [b, h, q, k]

        context = torch.matmul(attention, V)  # [b, h, q, d]
        context = rearrange(context, 'b h q d -> b q (h d)')  # Concat heads

        x = self.linear(context)  # [b, q, emb_dim]
        return x


class MLPBlock(nn.Module) :
    def __init__(self, emb_dim: int, forward_dim: int, dropout_ratio: float) :
        super().__init__()
        
        self.linear_1 = nn.Linear(emb_dim, forward_dim * emb_dim)
        self.linear_2 = nn.Linear(forward_dim * emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        
    def forward(self, x) :
        x = self.linear_1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x) 
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module) :
    def __init__(self, emb_dim: int, num_heads: int, forward_dim: int, dropout_ratio: float) :
        super().__init__()

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.msa = MultiHeadAttention(emb_dim, num_heads, dropout_ratio)
        self.mlp = MLPBlock(emb_dim, forward_dim, dropout_ratio)


    def forward(self, x) :
        input = x
        x = self.layer_norm(x)  # [bach_size, patch_num + 1, emb_dim]
        x = self.msa(x)  # [bach_size, patch_num + 1, emb_dim]
        x = x + input  # [bach_size, patch_num + 1, emb_dim]

        _x = self.layer_norm(x)  # [bach_size, patch_num + 1, emb_dim]
        _x = self.mlp(_x)  # [bach_size, patch_num + 1, emb_dim]
        x  = x + _x  # [bach_size, patch_num + 1, emb_dim]
        
        return x
        

class VIT(nn.Module) :
    def __init__(self, in_channel: int, img_size: int, patch_size: int, emb_dim: int,
                 n_layers: int, num_heads: int, forward_dim: int, dropout_ratio: float, n_classes: int) :
        super().__init__()
        
        '''
        # Base  Layer: 12, Hidden dim: 768, MLP dim: 3072, Heads: 12, Params: 86M
        # Large Layer: 24, Hidden dim: 1024, MLP dim: 4096, Heads: 16, Params: 307M
        # Huge  Layer: 32, Hidden dim: 1280, MLP dim: 5120, Heads: 16, Params: 632M
        '''

        self.embeding_module = image_embedding(in_channel, img_size, patch_size, emb_dim)
        self.encoder_modules = nn.ModuleList([EncoderBlock(emb_dim, num_heads, forward_dim, dropout_ratio) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, n_classes) 


    def forward(self, x) :
        # image embedding
        # x: [batch, channel, width, height]
        x = self.embeding_module (x)  # [bach_size, patch_num + 1, emb_dim]

        # transformer_encoder
        for enc in self.encoder_modules :
          x = enc(x)  # [bach_size, patch_num + 1, emb_dim]

         # cls_token output 
        x = x[:, 0, :]  # [batch_size, emb_dim]
        
        x = self.layer_norm(x)  # [batch_size, emb_dim]
        x = self.classification_head(x)  # [batch_size, n_classes]

        return x