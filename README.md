# Stochastic Tunneling Time Simulations

This repository contains Python programs to simulate **tunneling times** using **Nelson's stochastic mechanics**. The simulations compute the distribution of tunneling times for quantum particles crossing potential barriers by generating stochastic trajectories associated with the quantum wave function.

Two different physical situations are implemented:

1. **Wave packet tunneling through a square potential barrier**
2. **Tunneling between wells in a double square well potential using stationary states**

The programs use **NumPy**, **Numba**, and **multiprocessing** to accelerate the simulations of stochastic trajectories.

---

# Physical Background

In Nelson's stochastic quantization, the motion of a quantum particle is described by the stochastic differential equation

$$
dx = b(x,t)dt + \sqrt{\frac{\hbar}{m}}dW_t
$$

where

- $b(x,t) = u(x,t) + v(x,t)$ is the **forward drift**
- $u$ is the **osmotic velocity**
- $v$ is the **current velocity**
- $W_t$ is a **Wiener process**.

For a wave function

$$
\Psi = \sqrt{\rho} e^{iS/\hbar}
$$

the velocities are

$$
u = \frac{\hbar}{2m}\nabla \ln \rho
$$

$$
v = \frac{1}{m}\nabla S
$$

The programs simulate stochastic trajectories governed by these velocities and measure first-passage times, which are interpreted as tunneling times.

---

# Programs

## 1. Wave Packet Tunneling
**File**
Tunneling_time_dist-wave_packet-square_barrier-varV0.py

### Description

Simulates tunneling of a Gaussian wave packet through a square potential barrier for different barrier heights $V_0$.

The wave function is constructed as a superposition of stationary scattering states:

$$
\Psi(x,t) = \int g(k)\Psi_k(x,t)dk
$$

where $g(k)$ is the Gaussian momentum distribution of the initial wave packet.

For each barrier height:

1. The wave function is computed
2. The drift field $b(x,t)$ is constructed
3. Stochastic trajectories are simulated
4. The tunneling times are recorded.

### Output

A CSV file containing:
`V0_1, tau1, tau2, tau3, ...`
where `tau` are individual tunneling times.

### Key parameters

Wave packet parameters:

- `k_avg` – mean wave number
- `sigma` – momentum spread
- `x0_mean` – initial packet center

Barrier parameters:

- `a` – half barrier width
- `V0_MIN`, `V0_MAX`, `V0_STEP` – barrier heights

Simulation parameters:

- `sample` – number of trajectories per barrier
- `dt` – time step
- `X`, `T` – grids for drift evaluation

---

## 2. Stationary State Tunneling (Double Square Well)
**File**
Tunneling_time_dist-stationary_states.py

### Description

Simulates tunneling between wells in a double square well potential (DSW) using the ground state wave function.

In this case the drift field is time independent, since the system is in a stationary state, and equals the osmotic velocity $u(x)$.

The forward drift is given analytically in each region of the potential:

- Left well
- Barrier
- Right well

Stochastic trajectories are simulated starting in the left well until they reach the right well.

More details in: <https://arxiv.org/abs/2512.16168>

### Output

A CSV file containing the simulated tunneling times.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Physicist-Danilo/Stochastic_Tunneling_Times.git
cd Stochastic-Tunneling-Times
```

# Example Analysis in Python

```python
import numpy as np
data = np.loadtxt("times_DSW_10^6.csv", delimiter=",")
mean_time = np.mean(data)
```

```python
import numpy as np
with open('times_barrier_wave_packet_varV0.csv', 'r', encoding='utf-8') as f:
        content = f.readlines()
        A = [i.split(',') for i in content]
        V0 = []
        Times = []
        for i in A:
                V0.append(float(i[0]))
                Times.append(np.array([float(j) for j in i[1:]]))
```

# Author

Danilo Schafaschek

Federal University of Paraná (UFPR)

# License

This project is intended for academic and research use.

