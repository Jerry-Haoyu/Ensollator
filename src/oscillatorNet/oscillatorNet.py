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
        
        # The energy net G_e which parametrize the representation map, 
        # see the report for details
        self.G_e = nn.Sequential(
            nn.Linear(4, 32),
            self.__make_hidden__(hp['G_e_layers']),
            nn.Linear(32, self.d * 2)
        )
        
        # The frequency net G_f which defines the subspace
        # see the report for details
        self.G_f = nn.Sequential(
            nn.Linear(4, 32),
            self.__make_hidden__(hp['G_f_layers']), 
            nn.Linear(32, self.d)
        )
        
        self.mse = nn.MSELoss()
        
        # weights for data loss and physics loss
        self.lambda_data = 1.0
        self.lambda_physics = 1.0
        
    def __make_hidden__(self,num_hidden) -> nn.Module:
        hidden = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(32, 32), nn.Tanh()])
                for _ in range(num_hidden)
            ]
        )
        return hidden
    
    def forward(self, input_tensor):
        energies = self.G_e(input_tensor)
        freqs = self.G_f(input_tensor)
        # Note y(t) is batched, i.e., it gives output for y_i(t) for all i in the batch
        def y(t):
            cos_part = energies[:, :self.d] * torch.cos(freqs * t)
            sin_part = energies[:, self.d:] * torch.sin(freqs * t)
            return sin_part + cos_part
        return y
    
    def d_forward(self, input_tensor):
        energies = self.G_e(input_tensor)
        freqs = self.G_f(input_tensor)
        def dydt(t):
            cos_part = - freqs * energies[:, :self.d] * torch.sin(freqs * t)
            sin_part = freqs * energies[:, self.d:] * torch.cos(freqs * t)
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
        
        # data loss ∫(y-ỹ)^2 dt, see my report for detail
        data_loss = self.mse(y(ts), y_true)
        
        a_batch, b_batch, r_batch, delta_batch = input_tensor[:, 0].item(), input_tensor[:, 1].item(), input_tensor[:, 2].item(), input_tensor[:, 3].item()
        
        batch_size = input_tensor.shape[0]
        physics_loss = 0.0
        
        # this might be rather inefficient
        dydt_batch = dydt(ts)
        for batch_idx in range(batch_size):
            delta_step = int(delta_batch[batch_idx] / dt)
            bjerknes_term =  a_batch[batch_idx] * y_true[batch_idx, delta_step:] 
            # torch.roll(y_true, delta_step) shift the entire array right so at idex i we have ay[i]-by[i-delta_step]
            rossby_term = b_batch[batch_idx] * torch.roll(y_true[batch_idx], shifts=delta_step)[delta_step:]
            damp_term = r_batch[batch_idx] * y_true[batch_idx] ** 3
            # physics loss, see my report for detail
            physics_loss += torch.mean((dydt_batch[batch_idx, delta_step:] - (bjerknes_term - rossby_term - damp_term)) ** 2)
        
        physics_loss /= batch_size

        if adjust_weight:
            physics_loss_gradient = torch.autograd.grad(outputs=physics_loss, inputs=input_tensor, grad_outputs=torch.ones_like(physics_loss))
            data_loss_gradient = torch.autograd.grad(outputs=data_loss, inputs=input_tensor, grad_outputs=torch.ones_like(physics_loss))
            
            physics_loss_grad_norm = torch.norm(physics_loss_gradient, p=2).item()
            data_loss_grad_norm = torch.norm(data_loss_gradient, p=2).item()
            self.lambda_physics = 0.9 * self.lambda_physics + 0.1 * (physics_loss_grad_norm + data_loss_grad_norm) / physics_loss_grad_norm
            self.lambda_data = 0.9 * self.lambda_data + 0.1 * (physics_loss_grad_norm + data_loss_grad_norm) / data_loss_grad_norm
        
        total_loss = self.lambda_physics * physics_loss + self.lambda_data * data_loss
        
        return total_loss, physics_loss, data_loss, self.lambda_physics, self.lambda_data