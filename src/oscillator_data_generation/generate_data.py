"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""

import numpy as np
import os
import multiprocessing as mp 
import tqdm

from delayed_oscillator_solver import delayedOscillatorSolver as solver

### data_dir ###
out_dir_root = 'data'
################

def oscillation_check(alpha, delta_hat):
    """
        return true if not stable, i.e., oscillatory
    """
    if np.abs((3 * alpha - 2)/alpha) >= 1.0 :
        return False
    delta_hat_stab = np.sqrt((alpha ** 2 - (3 * alpha-2) ** 2)) * np.arccos((3 * alpha - 2)/alpha)
    return delta_hat > delta_hat_stab

def simulate(params):
    delta, a, b, r = params
    out_dir = os.path.join(out_dir_root, f"{delta:.0f}_{a:.3f}_{b:.3f}_{r:.3f}")
    model = solver(T0 = 2.0, 
                   out_dir=out_dir, 
                   delta=delta, 
                   a=a, 
                   b=b, 
                   r=r)
    model.simulate(save_fig=True)

if __name__ == "__main__":
    n_worers = mp.cpu_count()
    
    #### Sampling Density #####
    N_deltas = 30
    N_as = 30
    N_bs = 30
    N_rs = 30
    ###########################
    
    params_list = []
    N=0
    for delta in np.linspace(200.0, 500.0, N_deltas):
        for a in np.linspace(1e-2, 10.0, N_as):
            for b in np.linspace(1e-3, 1.0, N_bs):
                for r in np.linspace(1e-3, 1e-1, N_rs):
                    alpha = b/a 
                    delta_hat = a * delta
                    if oscillation_check(alpha, delta_hat):
                        N += 1
                        params_list.append((delta, a, b, r))
    
    with mp.Pool() as pool:
        list(tqdm.tqdm(pool.imap_unordered(simulate, params_list), total=len(params_list),desc='Delayed Oscillator Simulation Progress'))
    
        
    N_max = N_deltas * N_as * N_bs * N_rs
    print(f"Total number of valid training samples generated is {N}, with a valid rate of {N/N_max}%")