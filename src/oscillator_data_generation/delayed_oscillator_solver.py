"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.typing as npt
from collections.abc import Callable

class delayedOscillatorSolver:
    def __init__(self, 
                 out_dir : str, 
                 T0,
                 delta, 
                 a, 
                 b, 
                 r,
                 dt = 1e-1,
                 total_time = 3000
        ):
        """
        Solve delayed oscillator equation by simple Heun's method, i.e., second order Runge-Kutta
        Note this is an explict method by resolving implicty through a euler guess, i.e., 
        predictor-corrector method. See README.md for more details 
        
        Args:
            out_dir(str): the out directory 
            dt : the step size in foward step
            total_time : total simulation time 
        """
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        
        self.dt = dt
        self.total_steps = int((total_time + delta) / dt)
        
        self.a = a 
        self.b = b 
        self.delta = delta
        self.r = r 
        
        self.Ts = np.zeros(self.total_steps) # history of temperatures 
        
        self.i0 = int(self.delta / self.dt) 
        
        # set initial conditions
        self.Ts[:self.i0] = T0 * (np.random.randn()/1000 + 1.0)
  
        self.i = self.i0
        self.ts = np.linspace(0.0, float(total_time), self.total_steps, dtype=float)
        
    def _rhs(self, T_curr):
        return self.a * T_curr - self.b * self.Ts[self.i - 1 - self.i0] - self.r * (T_curr ** 3)
    
    def _get_euler_guess(self):
        return self.Ts[self.i-1] + (self._rhs(self.Ts[self.i-1])) * self.dt
    
    def _plot_and_save(self):
        fig, ax = plt.subplots()
        ax.plot(self.ts, self.Ts)
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        ax.set_title(rf"$a={self.a:.3f}$,$b={self.b:.3f}$,$r={self.r:.3f}$ $\delta={(self.delta):.0f}$")
        
        fig.savefig(os.path.join(self.out_dir, "time_series"))
        plt.close()
    
    def simulate(self, save_fig=True, save_data=True):
        while self.i< self.total_steps:
            T_curr = self.Ts[self.i-1]
            T_eg = self._get_euler_guess()
            dT_pred = self._rhs(T_curr = T_eg) 
            dT_corr = self._rhs(T_curr = T_curr) 
            dT = 0.5 * (dT_pred + dT_corr)
            self.Ts[self.i] = T_curr + dT * self.dt
            self.i += 1

        self.ts = self.ts[self.i0:]
        self.Ts = self.Ts[self.i0:]  
        
        data = self.Ts
        
        params = np.array([self.a, self.b, self.r, self.delta])
        
        dt = self.dt
        
        if save_data :
            np.savez(os.path.join(self.out_dir, "data"), 
                     params=params, 
                     data=data, 
                     dt=dt)
            
        if save_fig :
            self._plot_and_save()
    
if __name__ == "__main__":
    solver = delayedOscillatorSolver(out_dir="runs/delayed_oscillator/exp1",
                            total_time=3000,
                            T0 = 2.0,
                            delta=150,
                            a=1/80,
                            b=0.008,
                            r=0.02 
                            )
    solver.simulate()