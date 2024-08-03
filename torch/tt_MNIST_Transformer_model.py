from torch import einsum
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3,
                                bias = False)

    def forward(self, x):
        b, n, _, h = *x.shape, 2
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j',
                      q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d',
                     attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = Attention(32)
        self.norm1 = nn.LayerNorm(32)
        self.fc1 = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, x):
        out = nn.functional.relu(self.attention(self.norm1(x)) + x)
        out = nn.functional.relu(self.fc1(self.norm2(out)) + out)
        return out


class MNISTTransformer(nn.Module):
    def __init__(self, depth):
        super().__init__()
        image_size = 28
        patch_size = 7
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2
        self.to_patches = lambda x: rearrange(x,
                                              'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                              p1=patch_size, p2=patch_size)
        self.embedding = nn.Linear(patch_dim, 32)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, 32))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
        self.features = nn.Sequential()
        for i in range(depth):
            self.features.append(Transformer())
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        patches = self.to_patches(x)
        x = self.embedding(patches)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token,
                            '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        out = self.features(x)[:, 0]
        return self.classifier(out).flatten(1)

