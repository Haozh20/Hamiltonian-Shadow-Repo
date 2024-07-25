import time, math, cmath, argparse, random, os
import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford
from qiskit.quantum_info import Pauli
from tqdm import tqdm

import qiskit
I = qiskit.quantum_info.Pauli('I')
X = qiskit.quantum_info.Pauli('X')
Y = qiskit.quantum_info.Pauli('Y')
Z = qiskit.quantum_info.Pauli('Z')

# Parameters
global tol, x_dist
global data,snapshot
global n_idx,t_idx,H_idx

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-DM', type=int, default= 20000, help = 'datapool size')
    parser.add_argument('-M', type=int, default = 10000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 1000, help='number of estimators')
    parser.add_argument('-ntable', type=int, nargs='+', default = [3,6,8], help='qubit number table')

    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-tmin', type=int, default = 2)
    parser.add_argument('-tmax', type=int, nargs='+', default = [20,17,14,11,8,5])

    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')
    parser.add_argument('-xdist', type=float, default=9, help='atom distance for twirling Hamiltonian')
    parser.add_argument('-randamp', type=float, default=1, help='random factor in atom distance')

    parser.add_argument('-name', type=str, default='rydberg_fidelity', help = 'name of the data recorded')
    parser.add_argument('-obs', type=int, default = 0 , help = 'type of observable tested')
    parser.add_argument('-ens', type = int, default = 1, help = 'type of protocol, 1 for Hamiltonian shadow')

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

def I_str(n):
    res = I
    for i in range(n-1):
        res = res ^ I
    return res

def V_matrix(C, x_lst):
    n = len(x_lst)
    res = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            res[i][j] = C/(x_lst[i]-x_lst[j])**6
            res[j][i] = C/(x_lst[i]-x_lst[j])**6
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

    res = Omega_lst[0]/2*np.cos(phi_lst[0]) * X_str_lst[0] - Omega_lst[0]/2*np.sin(
        phi_lst[0]) * Y_str_lst[0] - Delta_lst[0]/2 * Z_str_lst[0]
    for i in range(n-1):
        res = res + Omega_lst[i+1]/2*np.cos(phi_lst[i+1]) * X_str_lst[i+1] - Omega_lst[i+1]/2*np.sin(
            phi_lst[i+1]) * Y_str_lst[i+1] - Delta_lst[i+1]/2 * Z_str_lst[i+1]

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
                    res = res + (Vij * (Z + I) ^ I_str(j-i-1)
                                 ^ (Z + I) ^ I_str(n-j-1))
            elif i == n-2:
                res = res + (Vij * I_str(n-2) ^ (Z + I) ^ (Z + I))
            else:
                if j == i+1:
                    #print((Vij* I_str(i) ^ (Z + I) ^ (Z + I) ^ I_str(n-j-1)))
                    res = res + (Vij * I_str(i) ^ (Z + I)
                                 ^ (Z + I) ^ I_str(n-j-1))
                elif j == n-1:
                    res = res + (Vij * I_str(i) ^ (Z + I)
                                 ^ I_str(j-i-1) ^ (Z + I))
                else:
                    res = res + (Vij * I_str(i) ^ (Z + I) ^
                                 I_str(j-i-1) ^ (Z + I) ^ I_str(n-j-1))

    return res.to_matrix()

# ensemble 0: Wrong shadow which is a hybird of Hamiltonian experiments and Global post-processing
def Wrong_Global_Shadow(rho, l, V, t_min, t_max):
    dim = len(rho)
    t = t_min+(t_max-t_min)*np.random.rand()
    # U = e^(-iHt)
    V_dag = V.conj().T
    ll = [cmath.exp(-1j*t*l[i]) for i in range(dim)]
    L = np.diag(ll)
    U = np.dot(V, np.dot(L, V_dag))
    U_dag = U.conj().T

    evoluted_rho = np.dot(U, np.dot(rho, U_dag))
    prob_lst = [0 for i in range(dim)]
    for i in range(dim):
        if abs(evoluted_rho[i][i]) > 10**(tol):
            prob_lst[i] = evoluted_rho[i][i].real
        else:
            prob_lst[i] = 0
    prob_lst = prob_lst / sum(prob_lst)
    measurement_res = np.random.choice(np.arange(0, dim), p=prob_lst)

    # calculate the post-measurement state
    rho_hat = np.outer(U[measurement_res,:].conj(),U[measurement_res,:])
    rho_hat = (dim+1)*rho_hat - np.identity(dim)
    return rho_hat

# ensemble 1: Our shadow using Hamiltonian
def Hamiltonian_Shadow(rho, l, V, X_mat, X_inv, t_min, t_max):
    dim = len(rho)
    t = t_min+(t_max-t_min)*np.random.rand()
    # U = e^(-iHt)
    V_dag = V.conj().T
    ll = [cmath.exp(-1j*t*l[i]) for i in range(dim)]
    L = np.diag(ll)
    U = np.dot(V, np.dot(L, V_dag))
    U_dag = U.conj().T

    evoluted_rho = np.dot(U, np.dot(rho, U_dag))
    prob_lst = [0 for i in range(dim)]
    for i in range(dim):
        if abs(evoluted_rho[i][i]) > 10**(tol):
            prob_lst[i] = evoluted_rho[i][i].real
        else:
            prob_lst[i] = 0
    prob_lst = prob_lst / np.sum(prob_lst)
    measurement_res = np.random.choice(np.arange(0, dim), p=prob_lst)

    # calculate the post-measurement state
    rho_hat = np.outer(U[measurement_res,:].conj(),U[measurement_res,:])

    # call the map of channel M^{-1}
    rho_hat = H_map(rho_hat, V, X_mat, X_inv)
    return rho_hat

# ensemble 2: correct global shadow protocol, use random unitary
def Original_Global_Shadow(rho):
    dim = len(rho)
    U = np.array(random_unitary(dim).to_matrix())

    evoluted_rho = np.dot(U, np.dot(rho, np.conj(U).T))
    prob_lst = [0 for i in range(dim)]
    for i in range(dim):
        if abs(evoluted_rho[i][i]) > 10**(tol):
            prob_lst[i] = evoluted_rho[i][i].real
        else:
            prob_lst[i] = 0
    prob_lst = prob_lst / sum(prob_lst)
    measurement_res = np.random.choice(np.arange(0, dim), p=prob_lst)

    rho_hat = np.outer(U[measurement_res,:].conj(),U[measurement_res,:])
    rho_hat = (dim+1)*rho_hat - np.identity(dim)
    return rho_hat

def Shadow_estimator(rho, obs, obs_type, DM, ensemble, l, V, X_mat, X_inv, t_min, t_max):
    res = 0

    DM_tqdm = tqdm(range(DM),leave=False)
    # DM_tqdm = range(DM)
    if (obs_type==0) or (obs_type==1) or (obs_type==3) or (obs_type==5): # Fidelity / Pauli / Hamiltonian / ZXZ
        
        # real_value = np.trace(np.dot(obs, rho))
        # ideal[ens_idx][n_idx][t_idx][H_idx] = real_value.real

        # if ensemble == 0:
        #     for i in DM_tqdm:
        #         rho_hat = Wrong_Global_Shadow(rho, l, V, t_min, t_max)
        #         term = np.trace(np.dot(obs, rho_hat))
        #         snapshot[n_idx][t_idx][H_idx][i] = term.real

        if ensemble == 1:
            for i in DM_tqdm:
                rho_hat = Hamiltonian_Shadow(rho, l, V, X_mat, X_inv, t_min, t_max)
                term = np.trace(np.dot(obs, rho_hat))
                snapshot[n_idx][t_idx][H_idx][i] = term.real

        # elif ensemble == 2:
        #     for i in DM_tqdm:
        #         rho_hat = Original_Global_Shadow(rho)
        #         term = np.trace(np.dot(obs, rho_hat))
        #         snapshot[n_idx][t_idx][H_idx][i] = term.real

    return

def performance(rho, obs, obs_type, DM, M, times, ensemble, l, V, X_mat, X_inv, t_min, t_max):

    Shadow_estimator(rho, obs, obs_type, DM, ensemble, l, V, X_mat, X_inv, t_min, t_max)
    
    pool = list(snapshot[n_idx][t_idx][H_idx])
    est_lst = [0]*times

    for times_idx in tqdm(range(times),leave=False):
        subset = random.sample(pool, M)
        est_lst[times_idx] = np.mean(subset)
    est_mean = np.mean(est_lst)
    est_std = np.std(est_lst)*np.sqrt(times/(times-1))

    data[H_idx][n_idx][t_idx][0] = est_mean
    data[H_idx][n_idx][t_idx][1] = est_std

    return


def Get_rho(n,obs_type):
    if obs_type==0: # fidelity
        
        # 01相间版-GHZ态
        dim = 2**n
        rho = np.zeros((dim,dim))
        b0 = 0
        b1 = 0
        for i in range(n):
            if i%2 == 0: # 第i位，i为偶数
                b1 += 2**i
            else: # 第i位，i为奇数
                b0 += 2**i
        rho[b0,b0] = 0.5
        rho[b0,b1] = 0.5
        rho[b1,b0] = 0.5
        rho[b1,b1] = 0.5

    return rho

# CHANGE: need changing when measure Hamiltonian
def Get_obs(obs_type,rho):
    n = int(math.log(len(rho), 2))
    
    if obs_type==0: # Fidelity
        obs = rho

    return obs

def Get_lvx(n, h_group_num, C0_ratio, x_dist, rand_amp):
    l_lst = []
    V_lst = []
    X_mat_lst = []
    X_inv_lst = []
    H_lst = []

    t0 = time.time()
    h_tqdm = tqdm(range(h_group_num),leave=False)
    for i in h_tqdm:
        C0 = 2*np.pi*(10)**6
        C = C0_ratio*C0
        x_lst = [x_dist*i+rand_amp*np.random.rand() for i in range(n)]
        # omega_lst = [1.1*2*np.pi + 0.5*np.random.rand()]*n
        omega_lst = [1.1*2*np.pi]*n
        delta_lst = [1.2*2*np.pi]*n
        phi_lst = [2.1]*n

        H = Rydberg_Hamiltonian(omega_lst, phi_lst, delta_lst, V_matrix(C, x_lst))
        l,V,X_mat,X_inv = VX_calculate(H)

        l_lst.append(l)
        V_lst.append(V)
        X_mat_lst.append(X_mat)
        X_inv_lst.append(X_inv)
        H_lst.append(H)
    
    t1 = time.time()
    print('n=',n,': finished (runtime:',t1-t0,')')
    
    return l_lst,V_lst,X_mat_lst,X_inv_lst,H_lst

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
ens_type = args.ens

DM = args.DM
M = args.M
times = args.times
t_max_lst = args.tmax
x_dist = args.xdist
rand_amp = args.randamp

obs_type = args.obs

t_num = len(t_max_lst)
n_table = args.ntable
n_num = len(n_table)

addr_data = "./store/data_"+args.name+".npy"
# addr_ideal = "./store/ideal_"+args.name+".npy"
create("./store/")


###############################  recording data & print info

# snapshot[n_idx][t_idx][H_idx][DM]
snapshot = np.zeros((n_num,t_num,h_group_num,DM))
# data[H_idx][n_idx][t_idx][mean / deviation]
data = np.zeros((h_group_num, n_num, t_num, 2))

print('DM=', DM, ', M=', M, ', times=', times, ', H_group_num=', h_group_num)
print('t_min=', t_min, ', t_max list=', t_max_lst)
print('atom distance=', x_dist)
print('n_table=', n_table)
if obs_type==0:
    print('**** Fidelity test:')
elif obs_type==5:
    print('**** ZXZ Cluster state test:')
print('--------------------------------')
print('')

############################# run

for n_idx in range(n_num):
    n = n_table[n_idx]
    print('--------------')
    print('*** n=',n)
    print('')

    # Get Hamiltonian H and then calculate l,V,X,X_inv, 改为直接从prep中读取
    if x_dist == int(x_dist):
        x_dist = int(x_dist)
    if rand_amp == int(rand_amp):
        rand_amp = int(rand_amp)
    if C0_ratio == int(C0_ratio):
        C0_ratio = int(C0_ratio)

    # prefix = '../prep/shadow_LVX/xdist'+str(x_dist)+'_rand'+str(rand_amp)+'_C'+str(C0_ratio)
    # l_lst = np.load(prefix +'/l_l/n'+str(n)+'.npy')
    # V_lst = np.load(prefix + '/V_l/n'+str(n)+'.npy')
    # X_mat_lst = np.load(prefix + '/X_mat_l/n'+str(n)+'.npy')
    # X_inv_lst = np.load(prefix + '/X_inv_l/n'+str(n)+'.npy')

    l_lst,V_lst,X_mat_lst,X_inv_lst,H_lst = Get_lvx(n, h_group_num, C0_ratio, x_dist, rand_amp)

    # Get state rho
    rho = Get_rho(n,obs_type)

    # Get observable obs, a list in case we measure Hamiltonian
    obs = Get_obs(obs_type, rho)

    # Start RUNNING!

    for H_idx in range(h_group_num):
        print('------')
        print('** Hamiltonian No.',H_idx,', n=', n, ':')
        print('')

        # ideal value
        real_value = 1

        for t_idx in range(t_num):
            t_max = t_max_lst[t_idx]
        
            t0 = time.time()
            performance(rho, obs, obs_type, DM, M, times, ens_type, l_lst[H_idx], V_lst[H_idx], X_mat_lst[H_idx], X_inv_lst[H_idx], t_min, t_max)
            t1 = time.time()
            
            print('* t in [', t_min, ', ', t_max, ']: bias=', data[H_idx][n_idx][t_idx][0] - real_value, ', standard deviation=', data[H_idx][n_idx][t_idx][1], ' (runtime=', t1-t0, 's)')
                

        np.save(addr_data, data)
        # np.save(addr_ideal, ideal)