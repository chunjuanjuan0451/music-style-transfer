from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim:  int = 89,
        hidden_dim: int = 256,
        z_p_dim:    int = 64,
        z_r_dim:    int = 64,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        z_dim = z_p_dim + z_r_dim  
        self.latent_to_hidden = nn.ModuleList([
            nn.Linear(z_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.rhythm_to_hidden = nn.Linear(1, hidden_dim)

    def _init_hidden(self, z: Tensor) -> Tensor:
        layers = [torch.tanh(fc(z)) for fc in self.latent_to_hidden]
        return torch.stack(layers, dim=0)  

    def _project_step_logits(self, step_logits: Tensor, temperature: float = 1.0) -> Tensor:
        scale = max(float(temperature), 1e-4)
        probs = torch.sigmoid(step_logits / scale)
        pitch_dim = self.input_dim - 1
        pitch_probs = probs[..., :pitch_dim]
        rest_prob = (1.0 - pitch_probs.amax(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        return torch.cat([pitch_probs, rest_prob], dim=-1)

    def forward(
        self,
        z_p:            Tensor,
        z_r:            Tensor,
        target:         Tensor,
        teacher_forcing: bool = True,
        rhythm_cond:    Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = target.shape
        z = torch.cat([z_p, z_r], dim=-1)   
        h = self._init_hidden(z)            

        sos = torch.zeros(batch_size, 1, self.input_dim, device=z.device)

        if teacher_forcing:
            dec_input = torch.cat([sos, target[:, :-1, :]], dim=1)  
            out, _ = self.gru(dec_input, h)                          
            if rhythm_cond is not None:
                out = out + self.rhythm_to_hidden(rhythm_cond.unsqueeze(-1))
            logits = self.output_proj(out)                          

        else:
            logits_list = []
            dec_input = sos   

            for _ in range(seq_len):
                out, h = self.gru(dec_input, h)          
                if rhythm_cond is not None:
                    bias_t = self.rhythm_to_hidden(rhythm_cond[:, _:_ + 1].unsqueeze(-1))
                    out = out + bias_t
                step_logits = self.output_proj(out)      
                logits_list.append(step_logits)
                pred = self._project_step_logits(step_logits)  
                dec_input = pred                         

            logits = torch.cat(logits_list, dim=1)       

        return logits

    def sample(
        self,
        z_p:         Tensor,
        z_r:         Tensor,
        seq_len:     int,
        temperature: float = 1.0,
        rhythm_cond: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            batch_size = z_p.size(0)
            z = torch.cat([z_p, z_r], dim=-1)
            h = self._init_hidden(z)

            sos = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
            dec_input = sos
            outputs = []

            for _ in range(seq_len):
                out, h = self.gru(dec_input, h)
                if rhythm_cond is not None:
                    bias_t = self.rhythm_to_hidden(rhythm_cond[:, _:_ + 1].unsqueeze(-1))
                    out = out + bias_t
                logits = self.output_proj(out)                    
                probs = self._project_step_logits(logits, temperature=temperature)
                outputs.append(probs)
                dec_input = probs

        return torch.cat(outputs, dim=1)   