# layers/PatchEmbedding.py
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding=0, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):                 # x: [B, L, D]
        B, L, D = x.shape
        # 滑动窗口把 L 切成 N 个 patch
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, N, D, P]
        x = x.permute(0, 1, 3, 2)        # [B, N, P, D]
        x = x.reshape(B, -1, self.patch_len)  # [B*N, P]
        x = self.value_embedding(x)      # [B*N, d_model]
        x = x.reshape(B, -1, x.size(-1))    # [B, N, d_model]
        x = self.dropout(x)
        return x                         # [B, N, d_model]