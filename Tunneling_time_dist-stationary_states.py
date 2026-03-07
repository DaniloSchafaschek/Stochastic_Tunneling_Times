import numpy as np
from numba import jit
import multiprocessing as mp
import os
import time

################ Definition of the forward drift ###############
#Here we define the forward drift D(x) for the double square well potential (DSW).

h_bar = 1      # Reduced Planck's constant (set to 1 for natural units)
m = 1          # Mass of the particle (set to 1 for natural units)

# DSW parameters:
d = 2.0        # Barrier width
a = d/2
b = 3.0        # Half of the well width
V0 = 2.0       # Potential barrier height
E = 0.746226   # Ground state energy level (obtained solving the Schrödinger equation)

k = np.sqrt(2*E)
kappa = np.sqrt(2*(V0-E))

# Forward drift for the DSW:
@jit(nopython=True) # @jit decorator from Numba compiles the function to machine code for faster execution.  
def D(x):
    if (-b < x <= -a):
        Osmotic_vel = (h_bar*k/m)*(1/np.tan(k*(x+b)))
    elif (-a < x <= a):
        Osmotic_vel = (h_bar*kappa/m)*np.tanh(kappa*x)
    elif (a < x < b):
        Osmotic_vel = (h_bar*k/m)*(1/np.tan(k*(x-b)))
    return Osmotic_vel
###############################################################

################# Simulation of trajectories ##################
@jit(nopython=True)
def simulate_trajectory(x0, t0, xc, Max_t, Min_x, dt):
    """
    Simulates a single trajectory of the stochastic process $dx = D(x)dt + dW$, where dW is a imcrement of the Wiener process and we are setting $\hbar = m = 1$. 
    The simulation continues until the trajectory reaches the threshold xc or exceeds the maximum time Max_t or maximum position Max_x.
    Returns the time T at which the trajectory reaches the threshold xc.

    :param x0: Initial position of the trajectory
    :param t0: Initial time of the trajectory
    :param xc: Threshold position
    :param Max_t: Maximum time allowed for the simulation
    :param Max_x: Maximum position allowed for the simulation (with opposite sign of the threshold)
    :param dt: Time step for the simulation
    """
    X0 = x0
    T0 = t0
    sqrt_dt = np.sqrt(dt)
    q = np.sqrt(h_bar/m)
    while True:
        dw = np.random.normal(0.0, sqrt_dt)
        X = X0 + D(X0)*dt + q*dw
        T = T0 + dt

        if X <= Min_x:
            X0 = x0
            T0 = t0
            continue

        if X >= xc:
            return T

        if T > Max_t:
            print("Trajectory exceeded maximum time without reaching the threshold.")
            return T
        
        X0 = X
        T0 = T

############# Parameters for the simulations ###############
sample = 1e6              # Total number of trajectories to simulate.
save_after = 1e4          # Number of trajectories to simulate before saving results to the file.
N = int(sample/save_after)
dt = 0.0001               # Time step for the simulations.
t0 = 0                    # Initial time for the trajectories.
x0 = -(a+b)/2             # Initial position for the trajectories (Here we start at the middle of the left well in the DSW).
xc = (a+b)/2              # Threshold position for the trajectories (Here we set the threshold at the middle of the right well in the DSW).
Max_t = 1e4               # Maximum time allowed for the simulation of a single trajectory.
Min_x = -b                # Minimum position allowed for the simulation (Here the trajectory is not allowed to go beyond the left infinity wall of the DSW).

file_name = 'times_DSW_10^6' # Name of the file where the results will be saved.
##############################################################

# Function to run a given number of simulations
def run_simulations(num_simulations):
    # Define a unique random seed for this execution
    np.random.seed(os.getpid())
    return [simulate_trajectory(x0, t0, xc, Max_t, Min_x, dt) for _ in range(num_simulations)]


if __name__ == "__main__":
    start_time = time.time() # Time of starting

    # Multiprocessing setup
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    with open(file_name + '.csv', 'wb') as file:
        for i in range(N):
            # Run simulations in parallel
            simulations_per_core = int(save_after/num_cores)
            results = pool.map(run_simulations, [simulations_per_core]*num_cores)

            # Concatenate results from all cores
            tau = [item for sublist in results for item in sublist]

            print(f"Batch {i+1}/{N}, Mean: {np.mean(tau)}")
            file.write((','.join(str(j) for j in tau) + ',').encode('utf-8'))
    
    pool.close()
    pool.join()

    file.close()

    end_time = time.time() # Time of ending

    print(f"Time of execution: {end_time - start_time} seconds") 
