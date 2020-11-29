from qutip import *
import numpy as np
import matplotlib.pyplot as plt

''' 
System parameters:

    SiV:
        N_states:           Relevant number of states, e.g. N_state = 4 if no magnetic field is applied (B=0)
        states:             List of states to consider, e.g. state[0]([1]) is lower(upper) ground state
        omega_i:            Transition frequency to state[i]
        d_avg:              Average dipole moment
        e:                  Elementary charge
        a_0:                Bohr radius

    Laser:
        e_0:                Vacuum permitivity
        c_0:                Vacuum speed of light
        P:                  Power
        w_0:                Gaussian beam waist    

    Laser-SiV:
        Omega_02(12):       Rabi frequency coupling lower(upper) ground to lower excited state with a Laser (described with a coherent state)
        delta_02(12):       Detuning of Laser 0(1) w.r.t. to transition frequency omega_2-omega_0(1)

    SiV-Environment:
        Gamma:              Matrix describing relaxation (diagonal) and dephasing (off-diagonal) rates, e.g.:
                            Gamma[1,1]: dephasing rate of state 1
                            Gamma[1,0]: relaxation rate of state 1 into state 0    
        c_ops:              Corresponding collapse operators

    External:
        T:                  Temperature
        B:                  Magnetic field
        theta_B:            Magnetic field orientation w.r.t. symmetry axis ([111])

    Interactions:           "Electronic Structure of the Silicon Vacancy Color Center in Diamond", C. Hepp, p. 77, eq. 2.90
        H_L02(12):          Laser-SiV interaction with rotating wave approximation
        H_ZS:               Zeeman interaction coupling spin
        H_ZL:               Zeeman interaction coupling to orbital momentum
        H_JT:               Jahn-Teller interaction coupling orbital momenta
        H_SO:               Spin-orbit interaction
        H_Strain:           Crystal strain
'''

# Constants
normalization = 1e-9
hbar = 1.054e-34
eps_0 = 8.85e-12
c_0 = 299792458
e = 1.602e-19
a_0 = 0.53e-10


''' System variables and operators:
    -Use arbirtray scaling factor for SiV optical dipole
    -Estimate for Rabi frequencies (Atom and Quantum Optics, p.186, eq. 5.186)
'''
N_states = 3         

state = []
for i in range(N_states):
    state.append(basis(N_states,i))                                          

sigma = []
sigma_tmp = []
for i in range(N_states):
    for j in range(N_states):
        sigma_tmp.append(state[i]*state[j].dag())
    sigma.append(sigma_tmp)

# Laser driving transition 0<->2
delta = 0
P = 10e-3
w_0 = 1e-6  
d_avg = 3*e*a_0                                                     
Omega = np.sqrt( (2*eps_0*c_0*(d_avg)**2*P)/(np.pi*hbar**2*w_0**2) )*normalization

# Environment
T = 2.7
gamma = 1/(2e-9)*normalization

Gamma_rel_down = gamma*(np.triu(np.ones((N_states,N_states)),k=0)-np.eye(N_states,N_states))
Gamma_rel_up = gamma*(np.tril(np.ones((N_states,N_states)),k=0)-np.eye(N_states,N_states))
Gamma_rel = Gamma_rel_down + Gamma_rel_up

Gamma_deph = 0.1*gamma*np.eye(N_states)

Gamma = Gamma_rel + Gamma_deph

print(Gamma)
c_ops = []
c_ops_tmp = []
for i in range(N_states):
    for j in range(N_states):
        c_ops_tmp.append(np.sqrt(Gamma[i,j])*sigma[i][j])
    c_ops.append(c_ops_tmp)


''' Lambda-System dynamics where hbar=1'''
H_0 = state[2][2]*delta
H_L = Omega*(state[0][2]+state[2][0])
H = H_0+H_L02+H_L12

''' Master-equation solver '''
# times = np.linspace(0.0, 100.0, 500)
# psi0 = state[0]
# options = Options()
# options.nsteps = 2000
# result = mesolve(H, psi0, times, c_ops, [sigma[2][2]])
# plt.plot(times,result.expect[0])
# plt.show()
