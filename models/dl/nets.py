"""
models/dl/nets.py —— 区域式(单模型全站, 静态属性入模)深度学习架构 + E6 五种损失
输入: x_dyn (B, T=365, D_dyn), x_stat (B, D_stat) -> yhat (B,) 当日流量(标准化空间)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------ 架构
class StaticEmbed(nn.Module):
    def __init__(self, d_stat, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_stat, d_out), nn.Tanh()) if d_stat > 0 else None
        self.d_out = d_out if d_stat > 0 else 0

    def forward(self, s):
        return self.net(s) if self.net is not None else None


class LSTMModel(nn.Module):
    """Kratzert 2019 EA-LSTM 简化版: 静态属性拼接到每个时间步 (concat-LSTM)."""
    def __init__(self, d_dyn, d_stat, hidden=256, layers=1, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(d_dyn + d_stat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, s):
        if s is not None and s.shape[1] > 0:
            x = torch.cat([x, s.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        o, _ = self.lstm(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


class GRUModel(LSTMModel):
    def __init__(self, d_dyn, d_stat, hidden=256, layers=1, dropout=0.4):
        nn.Module.__init__(self)
        self.lstm = nn.GRU(d_dyn + d_stat, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout); self.head = nn.Linear(hidden, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=2048):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TransformerModel(nn.Module):
    """原版 encoder-only Transformer (Vaswani 2017)."""
    def __init__(self, d_dyn, d_stat, d_model=128, nhead=8, layers=3, dropout=0.1, ff=256):
        super().__init__()
        self.inp = nn.Linear(d_dyn + d_stat, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, s):
        if s is not None and s.shape[1] > 0:
            x = torch.cat([x, s.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        h = self.enc(self.pos(self.inp(x)))
        return self.head(h[:, -1]).squeeze(-1)


class PatchTST(nn.Module):
    """Nie et al. 2023: 通道独立 + patch 化."""
    def __init__(self, d_dyn, d_stat, patch=16, stride=8, d_model=128, nhead=8, layers=3, dropout=0.1):
        super().__init__()
        self.patch, self.stride, self.d_dyn = patch, stride, d_dyn
        self.embed = nn.Linear(patch, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.stat = nn.Linear(d_stat, d_model) if d_stat > 0 else None
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(1))

    def forward(self, x, s):
        B, T, D = x.shape
        p = x.permute(0, 2, 1).unfold(-1, self.patch, self.stride)      # B,D,N,patch
        N = p.shape[2]
        h = self.embed(p.reshape(B * D, N, self.patch))
        h = self.enc(self.pos(h)).reshape(B, D, N, -1).mean(1)          # 通道平均
        if self.stat is not None and s is not None and s.shape[1] > 0:
            h = h + self.stat(s).unsqueeze(1)
        return self.head(h).squeeze(-1)


class TCN(nn.Module):
    """Bai et al. 2018 因果空洞卷积."""
    def __init__(self, d_dyn, d_stat, ch=64, levels=6, k=3, dropout=0.2):
        super().__init__()
        layers, cin = [], d_dyn + d_stat
        for i in range(levels):
            d = 2 ** i
            layers += [nn.Conv1d(cin, ch, k, padding=(k - 1) * d, dilation=d), nn.ReLU(), nn.Dropout(dropout)]
            cin = ch
        self.net = nn.Sequential(*layers); self.head = nn.Linear(ch, 1)

    def forward(self, x, s):
        if s is not None and s.shape[1] > 0:
            x = torch.cat([x, s.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        T = x.shape[1]
        h = self.net(x.transpose(1, 2))[:, :, :T]
        return self.head(h[:, :, -1]).squeeze(-1)


ARCHS = {"lstm": LSTMModel, "gru": GRUModel, "transformer": TransformerModel,
         "patchtst": PatchTST, "tcn": TCN}


# ------------------------------------------------------------------ E6 五种损失
def mse_loss(yhat, y, std=None, **kw):
    return torch.mean((yhat - y) ** 2)


def mae_loss(yhat, y, std=None, **kw):
    return torch.mean(torch.abs(yhat - y))


def nse_basin_loss(yhat, y, std=None, eps=0.1, **kw):
    """Kratzert 2019 basin-averaged NSE loss: sum (yhat-y)^2 / (std_basin+eps)^2."""
    return torch.mean((yhat - y) ** 2 / (std + eps) ** 2)


def logmse_loss(yhat, y, std=None, eps=1e-2, **kw):
    return torch.mean((torch.log(torch.clamp(yhat, min=0) + eps) - torch.log(torch.clamp(y, min=0) + eps)) ** 2)


def delta_loss(yhat, y, std=None, eta=1e-3, **kw):
    """本文 delta 损失 (Sec. 3.7): 分母加 eta*ybar 平滑."""
    ybar = torch.mean(torch.abs(y)) + 1e-6
    den = torch.abs(yhat) + torch.abs(y) + eta * ybar
    return torch.mean(torch.abs(yhat - y) / den)


LOSSES = {"mse": mse_loss, "mae": mae_loss, "nse": nse_basin_loss,
          "logmse": logmse_loss, "delta": delta_loss}
