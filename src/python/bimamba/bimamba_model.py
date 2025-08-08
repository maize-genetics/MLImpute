import torch
from torch import nn
from mamba_ssm.modules.mamba_simple import Mamba

# add citations

#########################################    DEFINE BIMAMBA ARCHITECTURE    ############################################

def create_block(
        d_model,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        device=None,
        dtype=None,
        d_conv=4,
):
    factory_kwargs = {"device": device, "dtype": dtype, "bidirectional": bidirectional, "bidirectional_strategy": bidirectional_strategy, "bidirectional_weight_tie": bidirectional_weight_tie, "d_conv": d_conv}
    mixer_cls = BiMambaWrapper(d_model=d_model, **factory_kwargs)
    norm_cls = nn.LayerNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
    return nn.Sequential(norm_cls, mixer_cls)

class BiMambaWrapper(nn.Module):
    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: str = "add",
            bidirectional_weight_tie: bool = True,
            d_conv = 4,
            **mamba_kwargs,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, d_conv=d_conv, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, d_conv=d_conv, **mamba_kwargs)
            if bidirectional_weight_tie:
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states):
        out = self.mamba_fwd(hidden_states)
        if self.bidirectional:
            out_rev = self.mamba_rev(hidden_states.flip(dims=(1,))).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return out

class BiMambaMixerModel(nn.Module):
    def __init__(self, d_model, n_layer, norm_epsilon=1e-5, bidirectional=True, bidirectional_strategy="add", d_conv=4, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    norm_epsilon=norm_epsilon,
                    bidirectional=bidirectional,
                    bidirectional_strategy=bidirectional_strategy,
                    layer_idx=i,
                    d_conv=d_conv,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
	)
        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

    def forward(self, input_features, output_hidden_states=False):
        all_hidden_states = []
        hidden_states = input_features.float()
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states)
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

class BiMamba(nn.Module):
    def __init__(self, d_model, n_layer, norm_epsilon=1e-5, bidirectional=True, bidirectional_strategy="add",
                 device=None, dtype=None, d_conv=4):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = BiMambaMixerModel(d_model, n_layer, norm_epsilon, bidirectional, bidirectional_strategy, d_conv=d_conv,
                                          **factory_kwargs)

    def forward(self, input_features, output_hidden_states=False, return_dict=False):
        hidden_states, all_hidden_states = self.backbone(input_features, output_hidden_states)
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states if output_hidden_states else None
            }
        return hidden_states, all_hidden_states if output_hidden_states else hidden_states


############################################    DEFINE LOSS FUNCTIONS    ###############################################

class SNPLoss(nn.Module):
    def __init__(self):
        super(SNPLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, unmasked_input, masked):
        targets = (unmasked_input[masked] > 0).to(torch.float32)
        return self.loss_fn(logits[masked], targets)  # (B, T, C)

class SNPLossSmooth(nn.Module):
    def __init__(self, lambda_smooth=0.2):
        super(SNPLossSmooth, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_smooth = lambda_smooth

    def forward(self, logits, unmasked_input, masked):
        diff = logits[:, 1:] - logits[:, :-1]
        smoothness_penalty = torch.mean(torch.abs(diff))
        targets = (unmasked_input[masked] > 0).to(torch.float32)
        return self.loss_fn(logits[masked], targets) + self.lambda_smooth * smoothness_penalty


class SNPLossSmoothAll(nn.Module):
    def __init__(self, lambda_smooth=0.2):
        super(SNPLossSmoothAll, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_smooth = lambda_smooth

    def forward(self, logits, unmasked_input, mask):
        targets = (unmasked_input > 0).to(torch.float32)
        diff = logits[:, 1:] - logits[:, :-1]
        smoothness_penalty = torch.mean(torch.abs(diff))
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return self.loss_fn(logits, targets) + self.lambda_smooth * smoothness_penalty



#################################################    DEFINE MODEL    ###################################################

class BiMambaSmooth(nn.Module):
    def __init__(self, input_dim, d_model, num_classes, n_layer, d_conv=4, device=None, dtype=None, lambda_smooth=0.2):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_projection = nn.Linear(input_dim, d_model, **factory_kwargs)
        self.bimamba = BiMamba(d_model, n_layer, d_conv=d_conv, **factory_kwargs)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes)
        )
        self.loss = SNPLossSmoothAll(lambda_smooth=lambda_smooth)

    def forward(self, input_features, hidden=False):
        with torch.cuda.amp.autocast():
            B, L, _ = input_features.shape # input_features: (B, L, input_dim)
            if self.training:
                mask = torch.rand(B, L, device=input_features.device) < 0.1 # randomly change 10% of input to 0 for training
                input_masked = input_features.masked_fill(mask.unsqueeze(-1), 0)
            else:
                # No masking during evaluation
                mask = torch.zeros(B, L, dtype=torch.bool, device=input_features.device)
                input_masked = input_features
            
            x = self.input_projection(input_masked)  # (B, L, d_model)
            hidden_states, _ = self.bimamba(x)  # (B, L, d_model)
            
            if hidden: return hidden_states
            predictions = self.classification_head(hidden_states)
            return predictions, mask


    def compute_loss(self, logits, unmasked, masked):
        return self.loss(logits, unmasked, masked)
