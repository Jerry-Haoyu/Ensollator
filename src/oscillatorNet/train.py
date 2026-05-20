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
                 hp = {'subspace_dim': 40, 'G_e_layers' : 3, 'G_f_layers' : 3},
                 note = None):
        
        # confirm device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(self.device == 'cpu') : raise RuntimeWarning("Device is now CPU !")
        
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

        # fixed set of sample indices for prediction visualization
        # drawn at random at init, then frozen for the rest of training
        num_vis_samples = min(10, len(self.dataset))
        self.fixed_sample_indices = np.random.choice(len(self.dataset), size=num_vis_samples, replace=False).tolist()
        
        # create model
        self.model = OscNet(hp).to(self.device)
        
        # create underlying grid 
        # length of time series
        self.dt = self.dataset.get_dt()
        self.total_steps = self.dataset.get_totalsteps()
        
        # get the grid (note all samples have the same grid)
        self.ts = torch.linspace(0.0, self.total_steps * self.dt, self.total_steps).to(self.device)
        
        # create lr scheduler and optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # logging
        with open(self.log_file, 'w') as f:
            print("=============== Meta Info - Exp Note ================", file=f)
            print(note, file=f)
            print("=====================================================", file=f)
            print("\n", file=f)
            print("=============== Model Hyperparameter ================", file=f)
            print(f"Subspace Dimension : {hp['subspace_dim']}", file=f)
            print(f"G_e number of layers : {hp['G_e_layers']}", file=f)
            print(f"G_f number of layers : {hp['G_f_layers']}", file=f)
            print("=====================================================", file=f)
            print("\n", file=f)
            print("=============== Train Hyperparameter ================", file=f)
            print(f"batch size : {batch_size}", file=f)
            print(f"total epochs : {epochs}", file=f)
            print(f"initial lr : {lr} \n adamW weight_decay : {weight_decay}", file=f)
            print(f"GPU: {torch.cuda.get_device_name(0)}", file=f)
            print("=====================================================", file=f)
        
        # training histories
        self.physics_loss_history_train = []
        self.data_loss_history_train = []
        self.physics_loss_history_test = []
        self.data_loss_history_test = []
        self.lambda_physics_history = []
        self.lambda_data_history = []
        self.learning_rate_history = []

        # create dashboard plot
        self.saved_dashboard_plot = os.path.join(self.output_dir, "dashboard_plot.png")
        fig, ax = plt.subplots(2,2)
        fig.suptitle(f"Dash Board For OscNet Trainer, Exp {max_n+1}")
        ax[0,0].set_xlabel("Epoch")
        ax[0,0].set_ylabel("Loss")
        ax[0,1].set_xlabel("Epoch")
        ax[0,1].set_ylabel(r"$\lambda$")
        ax[1,0].set_xlabel("Epoch")
        ax[1,0].set_ylabel("Loss")
        ax[1,1].set_xlabel("Epoch")
        ax[1,1].set_ylabel("Learning Rate")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        self.dash_board_plot = fig
        self.data_loss_plot_train, = ax[0,0].plot([], [], label=rf'$J_{{data}}^{{(train)}}$')
        self.data_loss_plot_test, = ax[0,0].plot([], [], label=rf'$J_{{data}}^{{(test)}}$')
        self.physics_loss_plot_train, = ax[1,0].plot([], [], label=rf'$J_{{physics}}^{{(train)}}$')
        self.physics_loss_plot_test, = ax[1,0].plot([], [], label=rf'$J_{{physics}}^{{(test)}}$')
        self.lambda_data_plot, = ax[0,1].plot([], [], label=rf'$\lambda_{{data}}$')
        self.lambda_physics_plot, = ax[0,1].plot([], [], label=rf'$\lambda_{{physics}}$')
        self.learning_rate_plot, = ax[1,1].plot([], [], label='lr')

        ax[0,0].legend(fontsize='small')
        ax[0,1].legend(fontsize='small')
        ax[1,0].legend(fontsize='small')
        ax[1,1].legend(fontsize='small')

        fig.savefig(self.saved_dashboard_plot)
        

    def __update_log(self, 
                     epoch,
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
        
        self.learning_rate_history.append(lr)
        
        log_line = (f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {float(train_loss):.3f}  | "
                    f"Test Loss: {float(test_loss):.3f} | "
                    f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
        print(log_line)
        with open(self.log_file, 'a') as f:
            print(log_line, file=f)
        
        if plot_dashboard == True:
            self.physics_loss_plot_train.set_data(np.arange(len(self.physics_loss_history_train)), self.physics_loss_history_train)
            self.physics_loss_plot_test.set_data(np.arange(len(self.physics_loss_history_test)), self.physics_loss_history_test)
            self.data_loss_plot_train.set_data(np.arange(len(self.data_loss_history_train)), self.data_loss_history_train)
            self.data_loss_plot_test.set_data(np.arange(len(self.data_loss_history_test)), self.data_loss_history_test)

            self.lambda_physics_plot.set_data(np.arange(len(self.lambda_physics_history)), self.lambda_physics_history)
            self.lambda_data_plot.set_data(np.arange(len(self.lambda_data_history)), self.lambda_data_history)
            
            self.learning_rate_plot.set_data(np.arange(len(self.learning_rate_history)), self.learning_rate_history)

            for ax in self.dash_board_plot.axes:
                ax.relim()
                ax.autoscale_view()

            # save the dashboard plot
            self.dash_board_plot.savefig(self.saved_dashboard_plot)
            
    def __plot_sample_predictions(self, epoch=None):
        """Plot ground truth vs model output for the fixed set of 10 samples on a 5x2 grid."""
        was_training = self.model.training
        self.model.eval()

        fig, axes = plt.subplots(5, 2, figsize=(12, 15))
        title = "Model Output vs Ground Truth"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        fig.suptitle(title)

        ts_np = self.ts.detach().cpu().numpy()

        with torch.no_grad():
            for plot_idx, sample_idx in enumerate(self.fixed_sample_indices):
                sample = self.dataset[sample_idx]
                params = sample['params'].unsqueeze(0).to(self.device)
                true_data = sample['data'].detach().cpu().numpy()

                y = self.model(params)
                pred = y(self.ts).squeeze(0).detach().cpu().numpy()

                ax = axes[plot_idx // 2, plot_idx % 2]
                ax.plot(ts_np, true_data, label='Ground Truth')
                ax.plot(ts_np, pred, label='Model Output', linestyle='--')

                a, b, r, delta = sample['params'].tolist()
                ax.set_title(f"a={a:.3f}, b={b:.3f}, r={r:.3f}, $\\delta$={delta:.3f}", fontsize=9)
                ax.set_xlabel("t")
                ax.set_ylabel("y(t)")
                if plot_idx == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        suffix = f"_epoch_{epoch}" if epoch is not None else ""
        save_path = os.path.join(self.output_dir, f"sample_predictions{suffix}.png")
        fig.savefig(save_path)
        plt.close(fig)

        if was_training:
            self.model.train()

    def __evaluate(self):
        self.model.eval() 
        total_loss : float = 0.0
        total_physics_loss : float = 0.0
        total_data_loss : float = 0.0
        for sample_batch in self.test_loader:
            sample_batch = sample_batch
            params_batch, data_batch = sample_batch['params'].to(self.device), sample_batch['data'].to(self.device)
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
            
        average_loss = total_loss / len(self.test_loader)
        average_physics_loss = total_physics_loss / len(self.test_loader)
        average_data_loss = total_data_loss / len(self.test_loader)

        return average_loss, average_physics_loss, average_data_loss

    def __train_one_epoch(self, adjust_weight=False):
        self.model.train() 
        total_loss : float = 0.0
        total_physics_loss : float = 0.0
        total_data_loss : float = 0.0
        for sample_batch in self.train_loader:
            sample_batch = sample_batch
            params_batch, data_batch = sample_batch['params'].to(self.device), sample_batch['data'].to(self.device)
            
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
        
        self.scheduler.step()
            
        average_loss = total_loss / len(self.train_loader)
        average_physics_loss =  total_physics_loss / len(self.train_loader)
        average_data_loss = total_data_loss / len(self.train_loader)
        
        return average_loss, average_physics_loss, average_data_loss, lambda_physics, lambda_data
    
    def train(self, T_log=10):
        total_start = time.time()
        best_error = float('inf')
        for epoch in range(self.epochs):
            epoch_start = time.time()

            average_loss, average_physics_loss, average_data_loss, \
            lambda_physics, lambda_data = self.__train_one_epoch(adjust_weight=(epoch % T_log == 0))

            average_test_loss, average_test_physics_loss, average_test_data_loss = self.__evaluate()

            epoch_time = time.time() - epoch_start
            
            lr = self.optimizer.param_groups[0]['lr']
            
            self.__update_log(epoch=epoch,
                              lr=lr, 
                              epoch_time=epoch_time,
                              train_loss=average_loss,
                              test_loss=average_test_loss,
                              train_ploss=average_physics_loss,
                              train_dloss=average_data_loss,
                              test_ploss=average_test_physics_loss,
                              test_dloss=average_test_data_loss,
                              lambda_physics=lambda_physics,
                              lambda_data=lambda_data,
                              plot_dashboard=(epoch % T_log == 0)
                              )
            
            if average_test_loss < best_error:
                best_error = average_test_loss
                torch.save(self.model, os.path.join(self.output_dir, 'oscNet_model.pt'))

            if epoch % T_log == 0:
                self.__plot_sample_predictions(epoch=epoch)

        # final snapshot at the end of training
        self.__plot_sample_predictions(epoch=self.epochs)

        total_time = time.time() - total_start
        print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    
            

if __name__ == "__main__":
    t = trainer(lr=1e-5,
                epochs=1000,
                data_dir="data", 
                output_dir="output/oscNet",
                batch_size=200,
                hp={'subspace_dim':40, 'G_e_layers':4, "G_f_layers":4},
                note="New architecture!, no load balancing and higher physics loss weight(+ its fixed)"
                )
    t.train()
        