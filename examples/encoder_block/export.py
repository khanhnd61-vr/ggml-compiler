"""A pre-norm transformer encoder block, compiled with
`ggml-compiler compile examples/encoder_block/export.py:program`."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ggml_compiler


class EncoderBlock(nn.Module):
    def __init__(self, d=256, heads=8, mlp=1024):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.ModuleList(nn.Linear(d, d) for _ in range(3))
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        q, k, v = (p(h).view(b, s, self.heads, d // self.heads).transpose(1, 2) for p in self.qkv)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(a.transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def program():
    torch.manual_seed(0)
    block = EncoderBlock().eval()
    return ggml_compiler.Program.single(block, (torch.randn(1, 64, 256),), input_names=["x"], output_names=["y"])
