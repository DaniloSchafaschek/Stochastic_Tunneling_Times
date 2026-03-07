import numpy as np
from numba import jit
import multiprocessing as mp
import os
import time

########## Parameters and constants of the wave packet and the potential barrier ##########

h_bar = 1                  # Reduced Planck's constant (set to 1 for natural units)
m = 1                      # Mass of the particle (set to 1 for natural units)
k_avg = 1.0                # Average wave number of the initial Gaussian wave packet
sigma = 0.1                # Standard deviation of the initial momentum distribution of the wave packet
x0_mean = -25              # Mean initial position of the wave packet
x0_std = h_bar/(2*sigma)   # Standard deviation of the initial position distribution of the wave packet (related to the momentum spread by the uncertainty principle)
a = 1/2                    # Half of the barrier width

V0_MIN = 0.05              # Initial potential barrier height to be simulated
V0_MAX = 5.0               # Final potential barrier height to be simulated
V0_STEP = 0.1              # Step size for the potential barrier heights to be simulated

X_MIN = x0_mean - 20       # Initial spatial position for evaluating the wave function and the drift.
X_MAX = a + 5              # Final spatial position for evaluating the wave function and the drift.
X_STEP = 0.05              # Step size for the spatial positions where the wave function and the drift will be evaluated.

T_MIN = 0                  # Initial time for evaluating the wave function and the drift.
T_MAX = 50                 # Final time for evaluating the wave function and the drift.
T_STEP = 0.05              # Step size for the time points where the wave function and the drift will be evaluated.

###########################################################################################

k_values = np.linspace(0.001, k_avg + 10*sigma, 500)  # Array of wave numbers to be used in the superposition of the wave function. 
                                                      # It cannot be negative, and it should cover a range that includes the average wave number of the initial wave packet
                                                      # and extends beyond it to capture the relevant dynamics.

V0_values = np.arange(V0_MIN, V0_MAX, V0_STEP)        # Array of potential barrier heights to be simulated

X = np.arange(X_MIN, X_MAX, X_STEP)                   # Array of spatial positions where the wave function and the drift will be evaluated.
T = np.arange(T_MIN, T_MAX, T_STEP)                   # Array of time points where the wave function and the drift will be evaluated.

# Lengths of the arrays X, T, and k_values.
NX = len(X)
NT = len(T)
NK = len(k_values)

###########################################################################################

# Normalized Gaussian distribution of momentum k representing the initial wave packet.
# The wave packet is centered around the average wave number k_avg, and its spread is determined by sigma.
# The initial position of the center of the wave packet is given by x0_mean.
Gk = ((1/(2*np.pi*(sigma**2)))**0.25)*np.exp(-((k_values-k_avg)**2)/(4*sigma**2))*np.exp(-1j*k_values*x0_mean)

####################### Functions used in the simulation #################################

@jit(nopython=True)
def Psi_barrier_vec(k : np.ndarray, V0, t):
    '''Calculates the wave function of a particle in a potential barrier for a given array of wave numbers k, potential barrier height V0, and time t.
    Returns a 2D array where each row corresponds to a wave number k[i] of k and each column corresponds to a spatial position x in the array X.'''

    E = (k**2/(2*m))*(h_bar**2)

    rho = np.where(V0 >= E, np.sqrt(2*m*(V0 - E))/h_bar, 1j*np.sqrt(2*m*(E - V0))/h_bar).astype(np.complex128)

    c1 = np.exp(2j*k*a)*np.cosh(2*rho*a)
    c2 = 0.5*((rho/k)-(k/rho))*np.exp(2j*k*a)*np.sinh(2*rho*a)
    c3 = 0.5*((rho/k)+(k/rho))*np.sinh(2*rho*a)

    alpha = -1j*c3/(c1 + 1j*c2)
    beta = 0.5*(1 + 1j*k/rho)*np.exp(-rho*a + 1j*k*a)/(c1 + 1j*c2)
    gamma = 0.5*(1 - 1j*k/rho)*np.exp(rho*a + 1j*k*a)/(c1 + 1j*c2)
    delta = 1/(c1 + 1j*c2)

    U = np.exp(-1j*E*t/h_bar)

    psi = np.empty((NK, NX), dtype=np.complex128)

    for i in range(NX):

        x_val = X[i]

        if x_val < -a:
            psi[:,i] = (np.exp(1j*k*x_val) + alpha*np.exp(-1j*k*x_val))*U

        elif x_val <= a:
            psi[:,i] = (beta*np.exp(rho*x_val) + gamma*np.exp(-rho*x_val))*U

        else:
            psi[:,i] = delta*np.exp(1j*k*x_val)*U

    return psi


@jit(nopython=True)
def Psi(t, V0):
    '''Calculates the total wave function at position x and time t for a given potential barrier height V0 by integrating over all wave numbers k'''
    dk = k_values[1] - k_values[0]
    psi_kx = Psi_barrier_vec(k_values, V0, t)
    Psi_vec = dk*np.dot(Gk, psi_kx)

    return Psi_vec


def B(t, V0):
    '''Calculates the drift velocity at time t for a given potential barrier height V0 by the formulas:
    u = (h_bar/m)*Re(Psi'/Psi) and v = (h_bar/m)*Im(Psi'/Psi), where Psi is the wave function and Psi' is its spatial derivative.'''
    Psi_vec = Psi(t, V0)
    Psi_vec_prime = np.gradient(Psi_vec, X)

    u = (h_bar/m)*np.real(Psi_vec_prime/Psi_vec)
    v = (h_bar/m)*np.imag(Psi_vec_prime/Psi_vec)

    return u + v


def compute_drift_matrix(V0):
    '''Create the drift matrix D for the potential barrier height V0'''
    D_matrix = np.zeros((NX, NT))
    for j, t in enumerate(T):
        D_matrix[:, j] = B(t, V0)
    return D_matrix


@jit(nopython=True)
def interp_D(D, x, t):
    '''Performs bilinear interpolation on the drift matrix D to find the drift velocity at a specific position x and time t.'''
    i = int((x - X[0])/(X[1]-X[0]))
    j = int((t - T[0])/(T[1]-T[0]))

    if i < 0:
        i = 0
    if i >= NX-1:
        i = NX-2

    if j < 0:
        j = 0
    if j >= NT-1:
        j = NT-2

    return D[i:i+2, j:j+2].mean()


@jit(nopython=True)
def simulate_trajectory(D, x0_mean, x0_std, t0, xc, dt, Max_t, Min_x):
    '''Simulates a single trajectory of a particle under the influence of the drift field D,
    starting from an initial position drawn from a normal distribution with mean x0_mean and standard deviation x0_std,
    and evolving in time with a time step dt until it either reaches the position xc (indicating tunneling through the barrier)
    or exceeds the maximum time Max_t or maximum position -Max_x (indicating reflection).
    The function returns the time spent by the particle from it first entering the barrier region (-a < x < a) until it reaches the position xc.
    '''
    X0 = np.random.normal(x0_mean, x0_std)
    while X0 >= -a:
        X0 = np.random.normal(x0_mean, x0_std)
    T0 = t0
    T_tunnel= 0
    sqrt_dt = np.sqrt(dt)
    q = np.sqrt(h_bar/m)
    while True:
        dw = np.random.normal(0.0, sqrt_dt)
        X = X0 + interp_D(D, X0, T0)*dt + q*dw
        T = T0 + dt
        
        if X <= Min_x or T >= Max_t:
            X0 = np.random.normal(x0_mean, x0_std)
            while X0 >= -a:
                X0 = np.random.normal(x0_mean, x0_std)
            T0 = t0
            T_tunnel = 0
            continue
        
        if T_tunnel > 0:
            T_tunnel += dt

        if X >= -a and T_tunnel == 0:
            T_tunnel = dt

        if X >= xc:
            return T_tunnel
        
        X0 = X
        T0 = T

############# Parameters for the simulations ###############

sample = 1e4    # Number of trajectories to be simulated for each potential barrier height.
dt = 0.01       # Time step for the trajectory simulation.
t0 = 0          # Initial time for the trajectory simulation.
xc = a          # Position beyond that indicates that the particle has successfully tunneled through the barrier.
Max_t = T[-1]   # Maximum time for the trajectory simulation.
Min_x = X[0]    # Minimum position for the trajectory simulation.

file_name = 'times_barrier_wave_packet_varV0' # Name of the file where the results will be saved.

###########################################################

# Function to run a given number of simulations
def run_simulations(D, num_simulations):
    np.random.seed(os.getpid()) # Define a unique seed for this process
    return [simulate_trajectory(D, x0_mean, x0_std, t0, xc, dt, Max_t, Min_x) for _ in range(num_simulations)]

if __name__ == "__main__":
    start_time = time.time() # Time of starting

    # Multiprocessing setup
    num_cores = mp.cpu_count() 
    pool = mp.Pool(num_cores)

    with open(file_name + '.csv', 'wb') as file:
        for V0 in V0_values:
            D_matrix = compute_drift_matrix(V0)

            # Run simulations in parallel
            simulations_per_core = int(sample/num_cores)
            results = pool.starmap(run_simulations, [(D_matrix, simulations_per_core)]*num_cores)

            # Concatenate results from all cores
            tau = [item for sublist in results for item in sublist]

            print(f"Potential energy: {round(V0, 2)} (Final potential energy: {round(V0_values[-1], 2)}), Mean tunneling time: {np.mean(tau)}")
            file.write((str(V0) + ',' + ','.join(map(str, tau)) + '\n').encode('utf-8'))
    
    pool.close()
    pool.join()

    file.close()

    end_time = time.time() # Time of ending

    print(f"Tempo de execução: {end_time - start_time} segundos")
