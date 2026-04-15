"""
Pure-PyTorch drop-in shim for mamba_ssm.Mamba.
Placed at venv/Lib/site-packages/mamba_ssm/__init__.py by install.bat.
No CUDA Toolkit or MSVC required. Slower than the real CUDA extension,
but produces correct output on CPU and GPU alike.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        fk = {"device": device, "dtype": dtype}

        self.in_proj  = nn.Linear(self.d_model,  self.d_inner * 2, bias=bias, **fk)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias, **fk,
        )
        self.act      = nn.SiLU()
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **fk)
        self.dt_proj  = nn.Linear(self.dt_rank,  self.d_inner, bias=True, **fk)
        self.out_proj = nn.Linear(self.d_inner,  self.d_model, bias=bias, **fk)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **fk)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        b, l, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = self.act(self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2))

        A  = -torch.exp(self.A_log.float())
        xp = self.x_proj(x)
        dt, B, C = xp.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        dA   = torch.exp(dt.unsqueeze(-1) * A)
        dB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        h  = torch.zeros(b, self.d_inner, self.d_state, device=hidden_states.device, dtype=hidden_states.dtype)
        ys = []
        for i in range(l):
            h  = dA[:, i] * h + dB_u[:, i]
            ys.append((h * C[:, i].unsqueeze(1)).sum(-1))

        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * self.act(z))


class MambaConfig:
    pass


def selective_scan_fn(*args, **kwargs):
    raise RuntimeError("selective_scan_fn not available in pure-PyTorch shim.")


__all__ = ["Mamba", "MambaConfig", "selective_scan_fn"]
