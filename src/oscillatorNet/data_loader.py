"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""
import torch 
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class oscNetData(Dataset):
    def __init__(self, data_dir):
        num_data = 0
        good_data = 0
        self.all_data = []
        self.dt = -1.0
        self.total_steps = -1 # length of each data
        for sample_dir in os.listdir(data_dir):
            sample_dir = os.path.join("data", sample_dir)
            if os.path.isdir(sample_dir):
                num_data += 1
                npz_file_path = os.path.join(sample_dir, "data.npz")
                try:
                    data_np = np.load(npz_file_path)
                    # Trigger a read to force the ZIP extraction test
                    _ = data_np['params'] 
                    
                    sample = {
                        "params" : torch.tensor(data_np['params'], dtype=torch.float32),
                        "data" : torch.tensor(data_np['data'], dtype=torch.float32),
                    }
                    
                    if (data_np['dt'] != self.dt and self.dt != -1): # i.e. dt is already initialized and differs with data_np['dt']
                        raise RuntimeWarning('Grid is inconsistent, samples have different dt')
                    
                    if (len(data_np['data']) != self.total_steps and self.total_steps != -1): # i.e. total_steps is already initialized and differs with time series length
                        raise RuntimeWarning('Grid is inconsistent, samples have different total_length(length of timeseries)')
                    
                    self.dt = data_np['dt'] 
                    self.total_steps = len(data_np['data'])
                    self.all_data.append(sample)
                    good_data += 1
                except Exception as e:
                    print(f"Skipping corrupted file at {npz_file_path}. Error: {e}")

        if(good_data < num_data) : print(f"[oscNetData] ==WARNING== DataSet has a yield of {good_data/num_data}, i.e., some are corrupted")
        
        self.len = good_data

    def get_dt(self):
        return self.dt
    
    def get_totalsteps(self):
        return self.total_steps
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        return self.all_data[index]
    
# helper functions to test the implementation of the Dataset class
def inpsect_random_sample():
    ds = oscNetData(data_dir="data")
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"size of train_loader is {len(train_loader)}")
    print(f"size of test_loader is {len(test_loader)}")

    random_sample = next(iter(train_loader))
    print(f"𝚯: a={random_sample['params'][0][0]}, \
          b={random_sample['params'][0][1]}, \
          r={random_sample['params'][0][2]}, \
          delta={random_sample['params'][0][3]}")
    dt =  ds.get_dt()
    
    
    total_steps = ds.get_totalsteps()
    
    print(f"grid properties: dt={dt}, total_steps={total_steps}")
    
    plt.plot(np.linspace(0,total_steps * dt, total_steps), random_sample['data'][0].detach().numpy())
    plt.savefig("tmp/inpsect_data_loader_time_series.png")
    
    
if __name__ == "__main__":
    inpsect_random_sample()
    # datas=np.load("data/200_0.010_0.005_0.001/data.npz")
    # input = torch.tensor(datas['params'], requires_grad=True)
    # print(input)