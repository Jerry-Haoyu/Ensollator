"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""

import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import os
import re

from oscillatorNet import OscNet
from data_loader import oscNetData

class trainer():
    def __init__(self, 
                 data_dir, 
                 output_dir, 
                 epochs = 1000,
                 batch_size = 1,
                 lr=1e-3,
                 weight_decay=0.01,
                 hp = {'subspace_dim': 40, 'G_e_layers' : 3, 'G_f_layers' : 3}):
        
        # create a new ouput directory for this new experiment
        pattern = re.compile(r"^exp(\d+)$")
        max_n = -1
        for exp_dir in os.listdir(output_dir):
            match = pattern.match(exp_dir)
            if match:
                n = int(match.group(1))
                max_n = max(max_n, n)
        self.output_dir = os.path.join(output_dir, f"exp{max_n+1}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Creating output directory {self.output_dir}")
        self.log_file = os.path.join(self.output_dir, "training_log")
        
        # create data_loaders
        self.dataset = oscNetData(data_dir)
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [0.8, 0.2])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # create model
        self.model = OscNet(hp)
        
        # create underlying grid 
        # length of time series
        self.dt = self.dataset.get_dt()
        self.total_steps = self.dataset.get_totalsteps()
        
        # get the grid (note all samples have the same grid)
        self.ts = torch.linspace(0.0, self.total_steps * self.dt, self.total_steps)
        
        # create lr scheduler and optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # confirm device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(self.device == 'cpu') : raise RuntimeWarning("Device is now CPU !")
        
        # logging
        with open(self.log_file, 'w') as f:
            print("=============== Model Hyperparameter ================", file=f)
            print(f"Subspace Dimension : {hp['subspace_dim']}", file=f)
            print(f"G_e number of layers : {hp['G_e_layers']}", file=f)
            print(f"G_f number of layers : {hp['G_f_layers']}", file=f)
            print("=====================================================", file=f)
            print("\n\n", file=f)
            print("=============== Train Hyperparameter ================", file=f)
            print(f"batch size : {batch_size}", file=f)
            print(f"total epochs : {epochs}", file=f)
            print(f"initial lr : {lr};adamW weight_decay : {weight_decay}", file=f)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("=====================================================", file=f)
        
        # create dashboard plot
        self.saved_dashboard_plot = os.path.join(self.output_dir, "dashboard_plot.png")
        fig, ax = plt.subplots(2,2)
        fig.suptitle(f"Dash Board For OscNet Trainer, Exp {max_n+1}")
        ax[0,0].set_xlabel("Epoch")
        ax[0,0].set_ylabel("Loss")
        ax[0,1].set_xlabel("Epoch")
        ax[0,1].set_ylabel(r"$\lambda$")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.savefig(self.saved_dashboard_plot)
        
        self.dash_board_plot = fig
        self.data_loss_plot_train = ax[0,0].plot(label=rf'$J_{{data}}^{{(train)}}$')
        self.data_loss_plot_test = ax[0,0].plot(label=rf'$J_{{data}}^{{(test)}}$')
        self.physics_loss_plot_train = ax[0,0].plot(label=rf'$J_{{physics}}^{{(train)}}$')
        self.physics_loss_plot_test = ax[0,0].plot(label=rf'$J_{{physics}}^{{(test)}}$')
        self.lambda_data_plot = ax[0,1].plot([0],self.lambda_data_history, label=rf'$\lambda_{{data}}$')
        self.lambda_physics_plot = ax[0,1].plot([0],self.lambda_physics_history, label=rf'$\lambda_{{physics}}$')
        
        plt.legend()
        
        # training histories
        self.physics_loss_history_train = []
        self.data_loss_history_train = []
        self.physics_loss_history_test = []
        self.data_loss_history_test = []
        self.lambda_physics_history = [1.0]
        self.lambda_data_history = [1.0]
    
    def __update_log(self, 
                     epoch_time,
                     lr,
                     train_loss, 
                     test_loss, 
                     train_ploss,
                     train_dloss,
                     test_ploss,
                     test_dloss,
                     lambda_physics, 
                     lambda_data, 
                     plot_dashboard=False):
        
        self.physics_loss_history_train.append(train_ploss)
        self.physics_loss_history_test.append(test_ploss)
        self.data_loss_history_train.append(train_dloss)
        self.data_loss_history_test.append(test_dloss)
        self.lambda_physics_history.append(lambda_physics)
        self.lambda_data_history.append(lambda_data)
        
        with open(self.log_file, 'a') as f:
            print(f"Epoch {self.epochs+1:3d}/{self.epochs} | "
                f"Train Loss: {float(train_loss):.3f}  | "
                f"Test Loss: {float(test_loss):.3f} | "
                f"LR: {lr:.2e} | Time: {epoch_time:.1f}s", \
                file=f
                )
        
        if plot_dashboard == True:
            self.physics_loss_plot_train.set_xdata(np.arange(len(self.physics_loss_history_train)))
            self.physics_loss_plot_train.set_ydata(self.physics_loss_history_train)
            self.physics_loss_plot_test.set_xdata(np.arange(len(self.physics_loss_history_test)))
            self.physics_loss_plot_test.set_ydata(self.physics_loss_history_test)
            self.data_loss_plot_train.set_xdata(np.arange(len(self.data_loss_history_train)))
            self.data_loss_plot_train.set_ydata(self.data_loss_history_train)
            self.data_loss_plot_test.set_xdata(np.arange(len(self.data_loss_history_test)))
            self.data_loss_plot_test.set_ydata(self.data_loss_history_test)
            
            self.lambda_physics_plot.set_xdata(np.arange(len(self.lambda_physics_history)))
            self.lambda_data_plot.set_xdata(np.arange(len(self.lambda_data_history)))
            
            # save the dashboard plot
            self.dash_board_plot.savefig(self.saved_dashboard_plot)
            
    def __evaluate(self):
        self.model.eval() 
        total_loss : float = 0.0
        for sample_batch in self.test_loader:
            sample_batch = sample_batch.to(self.device)
            params_batch, data_batch = sample_batch['params'], sample_batch['data']
            dt = self.dataset.get_dt()
            
            loss, physics_loss, data_loss, lambda_physics, lambda_data = self.model.get_loss(
                input_tensor=params_batch, 
                y_true=data_batch, 
                dt=self.dt,
                ts=self.ts, 
                adjust_weight=False
            )
            
            total_loss += loss.item()
            total_physics_loss += physics_loss.item()
            total_data_loss += data_loss.item()
            
        average_loss = total_loss / len(self.train_loader)
        average_physics_loss =  total_physics_loss / len(self.train_loader)
        average_data_loss = total_data_loss / len(self.train_loader)
        
        return average_loss, average_physics_loss, average_data_loss
    
    def __train_one_epoch(self, adjust_weight=False):
        self.model.train() 
        total_loss : float = 0.0
        for sample_batch in self.train_loader:
            sample_batch = sample_batch.to(self.device)
            params_batch, data_batch = sample_batch['params'], sample_batch['data']
            dt = self.dataset.get_dt()
            
             
            loss, physics_loss, data_loss, lambda_physics, lambda_data = self.model.get_loss(
                input_tensor=params_batch, 
                y_true=data_batch, 
                dt=self.dt,
                ts=self.ts, 
                adjust_weight=adjust_weight
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            total_loss += loss.item()
            total_physics_loss += physics_loss.item()
            total_data_loss += data_loss.item()
            
        average_loss = total_loss / len(self.train_loader)
        average_physics_loss =  total_physics_loss / len(self.train_loader)
        average_data_loss = total_data_loss / len(self.train_loader)
        
        return average_loss, average_physics_loss, average_data_loss, lambda_physics, lambda_data
    
    def train(self):
        total_start = time.time()
        best_error = float('inf')
        for epoch in range(self.epochs):
            epoch_start = time.time()

            average_loss, average_physics_loss, average_data_loss, \
            lambda_physics, lambda_data = self.__train_one_epoch(adjust_weight=(epoch % 100 == 0))

            average_test_loss, average_test_physics_loss, average_test_data_loss = self.__evaluate()

            epoch_time = time.time() - epoch_start
            
            lr = self.optimizer.param_groups[0]['lr']
            
            self.__update_log(lr=lr, 
                              epoch_time=epoch_time,
                              train_loss=average_loss,
                              test_loss=average_test_loss,
                              train_ploss=average_physics_loss,
                              train_dloss=average_data_loss,
                              test_ploss=average_test_physics_loss,
                              test_dloss=average_test_data_loss,
                              lambda_physics=lambda_physics,
                              lambda_data=lambda_data,
                              plot_dashboard=(epoch % 100 == 0)
                              )
            
            if average_test_loss < best_error:
                best_error = best_error
                torch.save(self.model, os.path.join(self.output_dir, 'oscNet_model.pt'))
            
            # if epoch % 1000 == 0:
            #     x1, x2, l = self._evaluate_parametrized_functions()
            #     self._plot_pendulum(x1, x2, l, epoch)
        
        total_time = time.time() - total_start
        print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    
            

if __name__ == "__main__":
    t = trainer(data_dir="data", 
                output_dir="output/oscNet"
                )
    t.train()
        