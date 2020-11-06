from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

''' 
System parameters:

    SiV:
        N_orbs:           Relevant number of states, e.g. N_state = 4 if no magnetic field is applied (B=0)
        states:             List of states to consider, e.g. state[0]([1]) is lower(upper) ground state
        nu_i:            Transition frequency to state[i]
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
        delta_02(12):       Detuning of Laser 0(1) w.r.t. to transition frequency nu_2-nu_0(1)

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
hbar = 1.054e-34
h = hbar*2*np.pi
kB = 1.38e-23
eps_0 = 8.85e-12
c_0 = 299792458
e = 1.602e-19
a_0 = 0.53e-10
norm = 1e-9
mu_B = 14.1e9*h

''' System variables and operators:
    Orbitals ordering
    0: |e_gx>
    1: |e_gy>
    2: |e_ux>
    3: |e_uy>

    Therefore orbital plus operator L_p_x = |e_g_y><e_g_x| ... 
    Due to D3d symmetry only L_z is non-zero. L_x and L_y vanish.

    Define spin up |🠕> as lower energy state.
'''
N_orbs = 4 
e_g_x = basis(N_orbs, 0)
e_g_y = basis(N_orbs, 1)
e_u_x = basis(N_orbs, 2)
e_u_y = basis(N_orbs, 3)
L_p_g = e_g_y*e_g_x.dag()
L_m_g = L_p_g.trans()
L_p_e = e_u_y*e_u_x.dag()
L_m_e = L_p_e.trans()
L_z_g = hbar*(1j*L_m_g-1j*L_p_g)
L_z_e = hbar*(1j*L_m_e-1j*L_p_e)

N_spins = 2
s_up = basis(N_spins,0)
s_down = basis(N_spins,1)

S_m = s_up*s_down.dag()
S_p = s_down*s_up.dag()
S_x = hbar/2*(S_m + S_p)
S_y = -1j*S_m + 1j*S_p
S_z = s_down*s_down.dag() - s_up*s_up.dag()

N = N_orbs*N_spins

'''System Dynamics:
Orbital part comprising Spin-Orbit, Jahn Teller, Zeeman and Strain couplin
'''
HOrb = Qobj(np.zeros((N_orbs, N_orbs)))
HOrb += 1.68*e/h*(e_u_x*e_u_x.dag()+e_u_y*e_u_y.dag())

#SO-Coupling
SO_g = 46e9
SO_e = 250e9
HOrb += -(SO_g/2)*1j*e_g_x*e_g_y.dag()
HOrb += (SO_g/2)*1j*e_g_y*e_g_x.dag()
HOrb += -(SO_e/2)*1j*e_u_x*e_u_y.dag()
HOrb += (SO_e/2)*1j*e_u_y*e_u_x.dag()
H = tensor(HOrb, S_z)
print(H)


#Jahn-Teller-Coupling
JT_x_g = 0
JT_x_e = 0
JT_y_g = 0
JT_y_e = 0
HOrb = JT_x_g*e_g_x*e_g_x.dag()
HOrb = JT_y_g*e_g_x*e_g_y.dag()
HOrb = JT_y_g*e_g_y*e_g_x.dag()
HOrb = -JT_x_g*e_g_y*e_g_y.dag()
HOrb = JT_x_e*e_u_x*e_u_x.dag()
HOrb = JT_y_e*e_u_x*e_u_y.dag()
HOrb = JT_y_e*e_u_y*e_u_x.dag()
HOrb = -JT_x_e*e_u_y*e_u_y.dag()

H += tensor(HOrb, qeye(N_spins))
#Strain-Coupling
alpha_g = 0
beta_g = 0
alpha_e = 0
beta_e = 0
HOrb = alpha_g*e_g_x*e_g_x.dag()
HOrb = beta_g*e_g_x*e_g_y.dag()
HOrb = beta_g*e_g_y*e_g_x.dag()
HOrb = -alpha_g*e_g_y*e_g_y.dag()
HOrb = alpha_e*e_u_x*e_u_x.dag()
HOrb = beta_e*e_u_x*e_u_y.dag()
HOrb = beta_e*e_u_y*e_u_x.dag()
HOrb = -alpha_e*e_u_y*e_u_y.dag()

H += tensor(HOrb, qeye(N_spins))
print(H)


#Zeeman-Coupling
f = 0.1
gamma_S = 2*mu_B/hbar
gamma_L = mu_B/hbar
B_x = 0
B_y = 0
B_z = 0



H0[0,0] = gamma_S*B_z*e_g_x*e_g_y.dag()
H0[0,1] = gamma_S*(B_x-B_y*1j)*e_g_x*e_g_x.dag()
H0[0,2] = f*gamma_L*B_z*1j*e_g_x*e_g_x.dag()
H0[1,0] = gamma_S*(B_x+B_y*1j)
H0[1,1] = -gamma_S*B_z
H0[1,3] = f*gamma_L*B_z*1j
H0[2,0] = -f*gamma_L*B_z*1j
H0[2,2] = gamma_S*B_z
H0[2,3] = gamma_S*(B_x-B_y*1j)
H0[3,1] = -f*gamma_L*B_z*1j
H0[3,2] = gamma_S*(B_x+B_y*1j)
H0[3,3] = -gamma_S*B_z*1j


# ''' Driving part with rotating frame transformation'''         
# P = 1e-3
# w_0 = 2e-6  
# d_avg = 3*e*a_0  

# delta_20 = 0.0                                     
# HD = Qobj(np.zeros((N_orbs,N_orbs)))
# Omega = np.zeros((N_orbs, N_orbs))
# Omega[2][0] = np.sqrt( (2*eps_0*c_0*(d_avg)**2*P)/(np.pi*h**2*w_0**2) )
# Omega[2][0] = 0.1e9
# Omega[0][2] = Omega[2][0]
# Omega = Omega*norm

# print("Rabi Freq", np.sqrt( (2*eps_0*c_0*(d_avg)**2*P)/(np.pi*h**2*w_0**2) )*norm)
# nu[2][0] = nu[2][2] - delta_20

# for i in range(N_orbs):
#     for j in range(N_orbs):
#         HD += Omega[i][j]*state[i]*state[j].dag()
# HD = Qobj(HD)
# H0 -= nu[2][0]*state[2]*state[2].dag()


# ''' Environmental variables:
#     Use equal relaxation Gamma_rel and dephasing Gamma_deph rates
# '''
# T = 6
# Gamma = np.zeros((N_orbs,N_orbs))

# #Dephasing rates
# Gamma[1,1] = 0
# Gamma[2,2] = 0*200e6/(2*np.pi)
# Gamma[3,3] = 0

# #Optical relaxation rates
# Gamma[0,2] = 1/(1.7e-9)
# Gamma[1,2] = 1/(1.7e-9)

# #Phononic relaxation rates
# Gamma[0,1] = 1/(80e-9)
# Gamma[1,0] = Gamma[0,1]*(1/(np.exp(h*nu[1][1]/(norm*kB*T))-1))/(1/(np.exp(h*nu[1][1]/(norm*kB*T))-1)+1)
# Gamma[2,3] = 1/(0.4e-9)
# Gamma[3,2] = Gamma[2,3]*(1/(np.exp(h*(nu[3,3]-nu[2,2])/(norm*kB*T))-1))/(1/(np.exp(h*(nu[3,3]-nu[2,2])/(norm*kB*T))-1)+1)

# Gamma = Gamma*norm

# c_ops = []
# c_ops_tmp = []
# for i in range(N_orbs):
#     for j in range(N_orbs):
#         c_ops.append(np.sqrt(Gamma[i,j])*state[i]*state[j].dag())


# '''Pulse Sequence'''
# args = {'N':20, 'T':100e-9/norm, 'tau_incr':5e-9/norm}
# def pulse_seq(t, args):
#     for n in range(1,args['N']+1):
#         if n*args['T'] + np.sum(np.linspace(1,n-1,n-1))*args['tau_incr'] < t and t < n*args['T'] + np.sum(np.linspace(1,n,n))*args['tau_incr']: 
#             return 0
#     return 1

# H = [H0, [HD,pulse_seq] ]

# t_steps = 2000
# t_max = args['N']*args['T']+args['tau_incr']*np.sum(np.linspace(1,args['N'],args['N']))
# times = np.linspace(0.0, t_max, t_steps)
# rho0 = 1/(1+np.exp(-h*nu[1,1]/(norm*kB*T))+np.exp(-h*nu[2,2]/(norm*kB*T))+np.exp(-h*nu[3,3]/(norm*kB*T)))*(
#         np.exp(-h*nu[0,0]/(norm*kB*T))*state[0]*state[0].dag()
#         +np.exp(-h*nu[1,1]/(norm*kB*T))*state[1]*state[1].dag()
#         +np.exp(-h*nu[2,2]/(norm*kB*T))*state[2]*state[2].dag()
#         +np.exp(-h*nu[3,3]/(norm*kB*T))*state[3]*state[3].dag())
# print("Initial state")
# print(rho0)
# print(Gamma)

# ''' Master-equation solver '''
# options = Options()
# options.nsteps = 2000
# result = mesolve(H, rho0, times, c_ops, [state[0]*state[0].dag(),state[1]*state[1].dag(),state[2]*state[2].dag(),state[3]*state[3].dag()], args=args)
# rc('text', usetex=True)
# fig, axs = plt.subplots(2,1,True)
# axs[0].plot(times, [pulse_seq(t,args) for t in times], '-k', linewidth=1, alpha=0.3)
# axs[1].plot(times, result.expect[0], '-b', linewidth=1, alpha=0.5, label=r'$1$')
# axs[1].plot(times, result.expect[1], '-y', linewidth=1, alpha=0.3, label=r'$2$')
# axs[1].plot(times, result.expect[2], '-r', linewidth=1, alpha=1, label=r'$3$')
# axs[1].plot(times, result.expect[3], '-m', linewidth=1, alpha=0.3, label=r'$4$')
# axs[1].legend(loc=0)
# plt.show()