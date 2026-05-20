"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""

import torch
import torch.nn as nn

class OscNet(nn.Module):
    def __init__(self, hp = {'subspace_dim': 40, 'G_e_layers' : 3, 'G_f_layers' : 3}):
        super().__init__() 
        
        self.d = hp['subspace_dim']

        # The frequency net G_f which defines the subspace
        # see the report for details.
        # G_f is executed first: it learns the subspace to project into.
        self.G_f = nn.Sequential(
            nn.Linear(4, 48),
            self.__make_hidden__(hp['G_f_layers']),
            nn.Linear(48, self.d)
        )

        # The energy net G_e which parametrize the representation map,
        # see the report for details.
        # G_e is executed second, conditioned on the original input
        # (dim 4) concatenated with the frequencies produced by G_f
        # (dim self.d), so it learns the representation on the
        # subspace already chosen by G_f.
        self.G_e = nn.Sequential(
            nn.Linear(4 + self.d, 48),
            self.__make_hidden__(hp['G_e_layers']),
            nn.Linear(48, self.d * 2)
        )
        
        self.mse = nn.MSELoss()
        
        # weights for data loss and physics loss
        self.lambda_data = 1.0
        self.lambda_physics = 2.0
        
    def __make_hidden__(self,num_hidden) -> nn.Module:
        hidden = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(48, 48), nn.Tanh()])
                for _ in range(num_hidden)
            ]
        )
        return hidden
    
    def forward(self, input_tensor):
        freqs_raw = self.G_f(input_tensor)
        energies = self.G_e(torch.cat([input_tensor, freqs_raw], dim=-1)).unsqueeze(-1)
        freqs = freqs_raw.unsqueeze(-1)
        # print(f"\n\n\n shape of freqs is {freqs.shape} \n\n\n")
        # Note y(t) is batched, i.e., it gives output for y_i(t) for all i in the batch
        def y(t):
            t = t[None,None,:]
            # (B * d * 1) * (1 * 1 * len) -> (B * d * len), i.e. for each batch, we have a series expansion in the column for each t
            # print(f"\n\n\n shape of t is {t.shape} \n\n\n")
            cos_part = torch.sum(energies[:, :self.d] * torch.cos(t * freqs), dim=1)
            sin_part = torch.sum(energies[:, self.d:] * torch.sin(t * freqs), dim=1)
            return sin_part + cos_part
        return y
    
    def d_forward(self, input_tensor):
        freqs_raw = self.G_f(input_tensor)
        energies = self.G_e(torch.cat([input_tensor, freqs_raw], dim=-1)).unsqueeze(-1)
        freqs = freqs_raw.unsqueeze(-1)
        def dydt(t):
            t = t[None,None,:]
            cos_part = - torch.sum(freqs * energies[:, :self.d] * torch.sin(t * freqs), dim=1)
            sin_part = torch.sum(freqs * energies[:, self.d:] * torch.cos(t * freqs), dim=1)
            return sin_part + cos_part
        return dydt
        
    
    def get_loss(self, input_tensor, y_true, ts, dt, adjust_weight=False):
        """
        Args:
            a,b,r,delta : oscillator parameters Theta
            y_true: a batch of arrays that records the value of true y solved for the oscillator numerically in [0, t_max]]
            dt: an int that represents the time step used in the grid of the numerical solver used to generate data
            adjust_weight: whether to adjust the weight of the composite loss
        """
        # get the callable from the forward functions
        y = self.forward(input_tensor)
        dydt = self.d_forward(input_tensor)

        # evaluate model output and its analytical derivative once on the grid
        y_pred = y(ts)
        dydt_batch = dydt(ts)

        # data loss ∫(y-ỹ)^2 dt, see my report for detail
        data_loss = self.mse(y_pred, y_true)

        a_batch, b_batch, r_batch, delta_batch = input_tensor[:, 0].detach(), input_tensor[:, 1].detach(), input_tensor[:, 2].detach(), input_tensor[:, 3].detach()
        batch_size = input_tensor.shape[0]
        physics_loss = 0.0

        # PINN-style residual: RHS uses the model's own ỹ, matching the report
        for batch_idx in range(batch_size):
            delta_step = int(delta_batch[batch_idx].item()  / dt)
            bjerknes_term =  a_batch[batch_idx] * y_pred[batch_idx, delta_step:]
            # torch.roll(y_pred, delta_step) shifts the series right so index i holds ỹ(t_i - δ) for i >= delta_step
            rossby_term = b_batch[batch_idx] * torch.roll(y_pred[batch_idx], shifts=delta_step)[delta_step:]
            damp_term = r_batch[batch_idx] * y_pred[batch_idx, delta_step:] ** 3
            physics_loss += torch.mean((dydt_batch[batch_idx, delta_step:] - (bjerknes_term - rossby_term - damp_term)) ** 2)

        physics_loss /= batch_size

        total_loss = self.lambda_physics * physics_loss + self.lambda_data * data_loss
        
        # if adjust_weight:
        #     params = list(self.parameters())
        #     physics_grads = torch.autograd.grad(physics_loss, params, retain_graph=True)
        #     data_grads = torch.autograd.grad(data_loss, params, retain_graph=True)

        #     physics_loss_grad_norm = (torch.sqrt(sum([physics_grad.pow(2).sum() for physics_grad in physics_grads])).item() / torch.abs(physics_loss)).item()
        #     data_loss_grad_norm = (torch.sqrt(sum([data_grad.pow(2).sum() for data_grad in data_grads])).item() / torch.abs(data_loss)).item()

        #     # loss balancing
        #     self.lambda_physics = 0.99 * self.lambda_physics + 0.01 * (physics_loss_grad_norm + data_loss_grad_norm) / physics_loss_grad_norm 
        #     self.lambda_data = 0.99 * self.lambda_data + 0.01 *  (physics_loss_grad_norm + data_loss_grad_norm) / data_loss_grad_norm
        
        
        return total_loss, physics_loss, data_loss, self.lambda_physics, self.lambda_data