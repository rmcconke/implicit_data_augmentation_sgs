import torch
import torch.nn as nn


class sgsPointwiseEigenframeMLP(nn.Module):
    def __init__(self, layers=[20], activation=nn.LeakyReLU):
        """
        Args:
            filter_width: filter width Delta (scalar, fixed for all inputs)
            layers: list of hidden layer widths, e.g. [20] or [128, 128, 128]
            activation: activation function class (Prakash uses LeakyReLU)
        """
        super().__init__()

        # 4 inputs: hat_lambda3, hat_omega1, hat_omega2, hat_omega3
        # 5 outputs: deviatoric components [dev_11, dev_22, 12, 13, 23] (dev_33 = -dev_11 - dev_22)
        mlp_layers = []
        in_dim = 4
        for width in layers:
            mlp_layers.append(nn.Linear(in_dim, width))
            mlp_layers.append(activation())
            in_dim = width
        mlp_layers.append(nn.Linear(in_dim, 5))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, inputs):
        """
        Args:
            inputs: (B, 14) if precomputed: [invariants (4), eigenvectors (9), G (1)]
                    (B, 9) if raw velocity gradients
        Returns:
            (B, 5) deviatoric stress tensor components [dev_xx, dev_yy, xy, xz, yz]
        """
        if inputs.shape[-1] == 14:
            # Precomputed eigenframe inputs
            invariants = inputs[..., :4]
            eigenvectors = inputs[..., 4:13].reshape(-1, 3, 3)
            G = inputs[..., 13]
            
        elif inputs.shape[-1] == 9:
            # Raw velocity gradients - compute eigenframe on the fly
            vel_grad_mat = inputs.reshape(-1, 3, 3)
            
            S = 0.5 * (vel_grad_mat + vel_grad_mat.transpose(-2, -1))
            W = 0.5 * (vel_grad_mat - vel_grad_mat.transpose(-2, -1))
            
            G = torch.sqrt((S * S).sum(dim=(-2, -1)) + (W * W).sum(dim=(-2, -1)))
            G = G.clamp(min=1e-10)
            
            eigenvalues, eigenvectors = torch.linalg.eigh(S)
            eigenvalues = eigenvalues.flip(-1)
            eigenvectors = eigenvectors.flip(-1)
            
            # Vorticity
            omega = torch.stack([
                vel_grad_mat[..., 2, 1] - vel_grad_mat[..., 1, 2],
                vel_grad_mat[..., 0, 2] - vel_grad_mat[..., 2, 0],
                vel_grad_mat[..., 1, 0] - vel_grad_mat[..., 0, 1],
            ], dim=-1)
            
            # Orient eigenvectors
            v1 = eigenvectors[..., :, 0]
            v3 = eigenvectors[..., :, 2]
            v1 = torch.where((v1 * omega).sum(dim=-1, keepdim=True) < 0, -v1, v1)
            v3 = torch.where((v3 * omega).sum(dim=-1, keepdim=True) < 0, -v3, v3)
            v2 = torch.linalg.cross(v3, v1)
            eigenvectors = torch.stack([v1, v2, v3], dim=-1)
            
            # Compute invariants
            omega_S1 = (omega * v1).sum(dim=-1)
            omega_S2 = (omega * v2).sum(dim=-1)
            omega_S3 = (omega * v3).sum(dim=-1)
            lambda3 = eigenvalues[..., 2]
            
            invariants = torch.stack([
                lambda3 / G,
                omega_S1 / G,
                omega_S2 / G,
                omega_S3 / G,
            ], dim=-1)
        
        else:
            raise ValueError(f"Expected input shape (..., 14) or (..., 9), got {inputs.shape}")
        
        # MLP forward pass: outputs 5 components [dev_11, dev_22, 12, 13, 23] in eigenframe
        out = self.mlp(invariants)  # (B, 5)
        
        # Build symmetric deviatoric tensor in eigenframe
        B = out.shape[0]
        tau_hat_S = torch.zeros(B, 3, 3, device=inputs.device, dtype=inputs.dtype)
        tau_hat_S[:, 0, 0] = out[:, 0]
        tau_hat_S[:, 1, 1] = out[:, 1]
        tau_hat_S[:, 2, 2] = -(out[:, 0] + out[:, 1])  # traceless: dev_33 = -dev_11 - dev_22
        tau_hat_S[:, 0, 1] = tau_hat_S[:, 1, 0] = out[:, 2]
        tau_hat_S[:, 0, 2] = tau_hat_S[:, 2, 0] = out[:, 3]
        tau_hat_S[:, 1, 2] = tau_hat_S[:, 2, 1] = out[:, 4]
        
        # Scale and rotate back to physical space
        scale =  (G ** 2)
        tau_S = scale[:, None, None] * tau_hat_S
        tau_mat = eigenvectors @ tau_S @ eigenvectors.transpose(-2, -1)
        
        # Extract 5 independent components [dev_xx, dev_yy, xy, xz, yz]
        return torch.stack([
            tau_mat[:, 0, 0],
            tau_mat[:, 1, 1],
            tau_mat[:, 0, 1],
            tau_mat[:, 0, 2],
            tau_mat[:, 1, 2],
        ], dim=-1)