import time, math, cmath, argparse, os
import numpy as np
import scipy as sp
from scipy.linalg import expm
import qiskit as qs
from qiskit.quantum_info import partial_trace as ptrace
from qiskit.quantum_info import random_unitary,random_clifford
from qiskit.opflow import I,X,Y,Z
from qiskit.quantum_info import Pauli
from tqdm import tqdm

# Parameters
global tol, x_dist
global data
global n_idx,t_idx,H_idx

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-ntable', type=int, nargs='+', default = [3,4,5,6,7,8], help='qubit number table')
    parser.add_argument('-M', type=int, default = 1000, help='number of rounds for one estimator')

    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-tmin', type=int, default = 2)
    parser.add_argument('-tmax', type=int, default = 20)

    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')
    parser.add_argument('-xdist', type=float, default=9, help='atom distance for twirling Hamiltonian')
    parser.add_argument('-randamp', type=float, default=1, help='random factor in atom distance')

    parser.add_argument('-name', type=str, default='frame_potential_qubitnum', help = 'name of the data recorded')

    return parser.parse_args()

# 通过给定的H计算出相应N^{-1} channel的(X^D)^{-1}
def VX_calculate(H):
    dim = len(H)
    n = int(math.log(len(H), 2))

    # diagonalize the Hamiltonian H
    l, V = np.linalg.eig(H)

    V_abs = np.zeros((dim,dim),dtype = float)
    for i in tqdm(range(dim),leave=False):
        for j in range(dim):
            V_abs[i,j] = np.abs(V[i,j])**2

    X_mat = np.dot(V_abs.T,V_abs)
    X_inv = np.linalg.inv(X_mat)
    return l, V, X_mat, X_inv

def H_map(Post_state, V, X_mat, X_inv):

    D = len(Post_state)
    V_dag = V.conj().T
    # get the input state for channel N^{-1}
    P = np.dot(V_dag, np.dot(Post_state, V))

    rho_hat = P / X_mat
    rho_hat_diag = np.dot(X_inv,np.diag(P))
    diag = np.arange(D)
    rho_hat[diag,diag] = rho_hat_diag

    rho_hat = np.dot(V, np.dot(rho_hat, V_dag))
    return rho_hat

def Ideal_FP(d,t):
    res = 0
    if t==1:
        res = d
    elif t==2:
        res = 2*(d**2) - d
    elif t==3:
        res = 6*(d**3) - 9*(d**2) + 4*d
    return res

def Frame_potential(t,l,t_min,t_max,M):
    dim = len(l)

    t_lst_1 = [0]*M
    t_lst_2 = [0]*M
    for i in range(M):
        t_lst_1[i] = t_min + (t_max-t_min)*np.random.rand()
        t_lst_2[i] = t_min + (t_max-t_min)*np.random.rand()

    potential = 0
    for i in tqdm(range(M), leave=False):
        for j in range(M):
            term = 0
            for m in range(dim):
                term += cmath.exp(1j*(t_lst_1[i]-t_lst_2[j])*l[m])
            term = abs(term)
            term = term**(2*t)
            potential += term
    potential = potential/(M*M)

    return potential

def Get_rho(n):
    # 随机纯态
    U0 = random_unitary(2**n).to_matrix()
    rho0 = np.diag([1]+[0]*(2**n-1))
    rho = np.dot(U0, np.dot(rho0, U0.conj().T))

    return rho

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)

########################################################################################
####################                                            ########################
####################           Starting from here               ########################
####################                                            ########################
########################################################################################


############################## receiving args
args = parse()

C0_ratio = args.Cratio
h_group_num = args.hnum
tol = args.tol

t_min = args.tmin

M = args.M
t_max = args.tmax
x_dist = args.xdist
rand_amp = args.randamp

n_table = args.ntable
n_num = len(n_table)

create("./store/")
addr_data = "./store/data_"+args.name+".npy"


###############################  recording data & print info

# data[H_idx][n_idx][k_idx]
data = np.zeros((h_group_num, n_num, 3))

print('M=', M, ', H_group_num=', h_group_num)
print('t_min=', t_min, ', t_max=', t_max)
print('atom distance=', x_dist)
print('random factor=', rand_amp)
print('n_table=', n_table)
print('t in [', t_min, ', ', t_max, ']')
print('**** frame potential test:')
print('--------------------------------')
print('')

############################# run

for n_idx in range(n_num):
    n = n_table[n_idx]

    # Get Hamiltonian H and then calculate l,V,X,X_inv, 改为直接从prep中读取
    if x_dist == int(x_dist):
        x_dist = int(x_dist)
    if rand_amp == int(rand_amp):
        rand_amp = int(rand_amp)
    if C0_ratio == int(C0_ratio):
        C0_ratio = int(C0_ratio)

    prefix = '../Rydberg/prep/shadow_LVX/xdist'+str(x_dist)+'_rand'+str(rand_amp)+'_C'+str(C0_ratio)
    l_lst = np.load(prefix +'/l_l/n'+str(n)+'.npy')
    V_lst = np.load(prefix + '/V_l/n'+str(n)+'.npy')
    X_mat_lst = np.load(prefix + '/X_mat_l/n'+str(n)+'.npy')
    X_inv_lst = np.load(prefix + '/X_inv_l/n'+str(n)+'.npy')

    # Select rho and observable
    rho = Get_rho(n)

    # print('* Ideal Frame potential: F1=', Ideal_FP(2**n,1), ', F2=',Ideal_FP(2**n,2), ', F3=', Ideal_FP(2**n,3))


    t0 = time.time()
    for i in tqdm(range(h_group_num),leave=False):
        data[i,n_idx,0] = Frame_potential(1,l_lst[i],t_min,t_max,M) / Ideal_FP(2**n,1)
        data[i,n_idx,1] = Frame_potential(2,l_lst[i],t_min,t_max,M) / Ideal_FP(2**n,2)
        data[i,n_idx,2] = Frame_potential(3,l_lst[i],t_min,t_max,M) / Ideal_FP(2**n,3)
    t1 = time.time()
    print('** n=', n, ': F_1=', np.median(data[:,n_idx,0]), ', F_2=', np.median(data[:,n_idx,1]), ', F_3=', np.median(data[:,n_idx,2]), ' (runtime=', t1-t0, 's)')

    np.save(addr_data, data)





