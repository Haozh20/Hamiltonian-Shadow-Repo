import time,cmath,math,argparse,random, os

import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford, random_hermitian
from qiskit.quantum_info import Pauli
from tqdm import tqdm
import matplotlib.pyplot as plt

I = qiskit.quantum_info.Pauli('I')
X = qiskit.quantum_info.Pauli('X')
Y = qiskit.quantum_info.Pauli('Y')
Z = qiskit.quantum_info.Pauli('Z')

# Parameters
global x_dist, C0_ratio

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-n', type=int, default = 12, help='the qubit number of the entire system')
    parser.add_argument('-t',type = int, nargs='+',default=[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2], help='list of time of all the smapshots')
    parser.add_argument('-xdist',type = float, default = 11, help = 'atom distance for evolution hamiltonian')
    parser.add_argument('-thpoints', type = int, default=1000, help = ' number of points in the theoretical result curve')
    parser.add_argument('-ckptnum', type=int, default = 10, help = 'number of checkpoints')
    parser.add_argument('-rho', type=int,default=1, help='type of initial state')
    

    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')
    # parser.add_argument('-istest')

    return parser.parse_args()

def I_str(n):
    res = I
    for i in range(n-1):
        res = res ^ I
    return res

# 一维Lattice情况
def old_V_matrix(C, x_lst):
    n = len(x_lst)
    res = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            res[i][j] = C/(x_lst[i]-x_lst[j])**6
            res[j][i] = C/(x_lst[i]-x_lst[j])**6
    return res

# 二维lattice情况
#xy_lst is a list of list, every list has two elements, standing for the x and y coordinates of a particle
def V_matrix(C,xy_lst): 
    n = len(xy_lst)
    res = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            res[i][j] = C/((xy_lst[i][0]-xy_lst[j][0])**2+(xy_lst[i][1]-xy_lst[j][1])**2)**3
            res[j][i] = C/((xy_lst[i][0]-xy_lst[j][0])**2+(xy_lst[i][1]-xy_lst[j][1])**2)**3
    return res

def Rydberg_Hamiltonian(Omega_lst, phi_lst, Delta_lst, v_matrix):
    V = v_matrix
    n = len(Omega_lst)
    # print('n=',n)
    X_str_lst = [X ^ I_str(n-1)]
    Y_str_lst = [Y ^ I_str(n-1)]
    Z_str_lst = [Z ^ I_str(n-1)]
    for i in range(n-2):
        X_str_lst.append(I_str(i+1) ^ X ^ I_str(n-i-2))
        Y_str_lst.append(I_str(i+1) ^ Y ^ I_str(n-i-2))
        Z_str_lst.append(I_str(i+1) ^ Z ^ I_str(n-i-2))
    X_str_lst.append(I_str(n-1) ^ X)
    Y_str_lst.append(I_str(n-1) ^ Y)
    Z_str_lst.append(I_str(n-1) ^ Z)

    res = Omega_lst[0]/2*np.cos(phi_lst[0]) * X_str_lst[0] - Omega_lst[0]/2*np.sin(phi_lst[0]) * Y_str_lst[0] - Delta_lst[0]/2 * Z_str_lst[0]
    for i in range(n-1):
        res = res + Omega_lst[i+1]/2*np.cos(phi_lst[i+1]) * X_str_lst[i+1] - Omega_lst[i+1]/2*np.sin(phi_lst[i+1]) * Y_str_lst[i+1] - Delta_lst[i+1]/2 * Z_str_lst[i+1]

    for i in range(n):
        for j in range(i+1, n):
            # print(i,j)
            Vij = V[i][j]/4
            if i == 0:
                if j == (j == i+1) & (j != n-1):
                    res = res + (Vij * (Z + I) ^ (Z + I) ^ I_str(n-2))
                elif j == n-1:
                    res = res + (Vij * (Z + I) ^ I_str(n-2) ^ (Z + I))
                else:
                    res = res + (Vij * (Z + I) ^ I_str(j-i-1) ^ (Z + I) ^ I_str(n-j-1))
            elif i == n-2:
                res = res + (Vij * I_str(n-2) ^ (Z + I) ^ (Z + I))
            else:
                if j == i+1:
                    #print((Vij* I_str(i) ^ (Z + I) ^ (Z + I) ^ I_str(n-j-1)))
                    res = res + (Vij * I_str(i) ^ (Z + I) ^ (Z + I) ^ I_str(n-j-1))
                elif j == n-1:
                    res = res + (Vij * I_str(i) ^ (Z + I) ^ I_str(j-i-1) ^ (Z + I))
                else:
                    res = res + (Vij * I_str(i) ^ (Z + I) ^ I_str(j-i-1) ^ (Z + I) ^ I_str(n-j-1))

    return res.to_matrix()

# get an evoluted subsystem
def Get_rho(n,t,H,rho_type):
    dim = 2**n

    # U = e^(-iHt)
    U = expm(-1j*t*H)
    U_dag = U.conj().T
    Rho0 = np.zeros((dim**2,dim**2))

    ## 0. b = 1111 0000
    if rho_type==0:
        b = 0
        for i in range(n):
            b += 2**(n+i)
        Rho0[b,b] = 1

    ## 1. b= 0000 1111
    elif rho_type==1:
        b = 0
        for i in range(n):
            b += 2**(i)
        Rho0[b,b] = 1
    
    ## 2. b= 0101 0101
    elif rho_type==2:
        b = 0
        for i in range(n):
            b += 2**(2*i+1)
        Rho0[b,b] = 1

    ## 3. GHZ tensor GHZ
    elif rho_type==3:
        rho0 = np.zeros((dim,dim))
        b0 = 0
        b1 = 0
        for i in range(n):
            if i%2 == 0: # 第i位，i为偶数
                b1 += 2**i
            else: # 第i位，i为奇数
                b0 += 2**i
        rho0[b0,b0] = 0.5
        rho0[b0,b1] = 0.5
        rho0[b1,b0] = 0.5
        rho0[b1,b1] = 0.5
        Rho0 = np.kron(rho0,rho0)
        

    # Rho_total = np.outer(U[:,b],U[:,b].conj())
    Rho_total = np.dot(U,np.dot(Rho0,U_dag))
    rho = np.trace(Rho_total.reshape(dim,dim,dim,dim),axis1=0,axis2=2)
    # print('trace:',np.trace(rho))

    return rho

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)

########################################################################################
####################                                            ########################
####################           Starting from here               ########################
####################                                            ########################
########################################################################################


### receiving args
args = parse()
n = int(args.n / 2) # 此处n为一半系统的qubit number
t_table = args.t
t_num = len(t_table)
theory_points = args.thpoints
x_dist = args.xdist
C0_ratio = args.Cratio
checkpoint_num = args.ckptnum
rho_type = args.rho

print('')
print('time list:',t_table)
print('qubit number of the entire system=',2*n)
print('C0_ratio=',C0_ratio, ', atom distance=',x_dist)
print('points in theory line=',theory_points)
print('**** Theoretical Entropy test:')
print('')

######################################## start theoretical running

theory_t_table = np.linspace(0,t_table[t_num-1],theory_points)
ent_data = np.zeros((theory_points))
# trace_data = np.zeros((theory_points))

# calculate H
n_total = 2*n
# get a Hamiltonian without randomness
C0 = 2*np.pi*(10)**6
C = C0_ratio*C0

# xy_lst = [[x_dist*i, 0] for i in range(n_total)]
xy_lst = [0]*n_total
for i in range(n):
    xy_lst[i] = [x_dist*i, 0]
    xy_lst[n+i] = [x_dist*i, x_dist]

omega_lst = [1.1*2*np.pi]*n_total
delta_lst = [1.2*2*np.pi]*n_total
phi_lst = [2.1]*n_total
H = Rydberg_Hamiltonian(omega_lst, phi_lst, delta_lst, V_matrix(C, xy_lst))

# select a snapshot, and analyze the system at this point
for t_idx in tqdm(range(theory_points),leave=False):
    t = theory_t_table[t_idx]

    rho = Get_rho(n,t,H,rho_type)

    trace_value = np.trace(np.dot(rho, rho)).real
    # real_value = - np.log2(trace_value)
    real_value = trace_value

    ent_data[t_idx] = real_value
    # print('** t=', round(t,4), ': trace=', trace_value, ', entropy=',real_value)

# save data
if x_dist==int(x_dist):
    x_dist = int(x_dist)

if rho_type==0:
    state = '_state1100'
elif rho_type==1:
    state = '_state0011'
elif rho_type==2:
    state = '_state0101'

addr = './store/theory_n' + str(2*n) + '_evoldist' + str(x_dist) + state + '_density'+ str(theory_points)

create("./store/")

np.save(addr+'.npy',ent_data)
print('------------------ Data recorded')

# draw figure

# plt.plot(theory_t_table, ent_data, marker='.', markersize=0, linewidth=1.5, linestype='-')
# plt.savefig(addr+'.png')