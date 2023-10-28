import time,math,cmath,argparse,random
import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford, random_hermitian
from qiskit.opflow import I, X, Y, Z
from qiskit.quantum_info import Pauli
from tqdm import tqdm

# Parameters
global tol,record,ideal
global ens_idx,n_idx,theta_idx,P_idx

def parse():
    parser = argparse.ArgumentParser(description='relationship of variance and qubit number')
    parser.add_argument('-DM', type=int, default = 10000, help='the size of the entire data pool')
    parser.add_argument('-M', type=int, default = 1000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 100, help='times of sampled subset from the datapool')
    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    
    parser.add_argument('-theta',type = float, nargs='+',default=[0.1,0.2,0.3], help='list of all the theta')
    parser.add_argument('-nlist',type = int, nargs='+',default=[2,3,4,5,6,7,8,9,10], help='list of all the n')

    parser.add_argument('-name', type=str, default='Var(n)_Fidelity', help = 'name of the data recorded')
    return parser.parse_args()

# 通过给定的H计算出相应N^{-1} channel的(X^D)^{-1}
def VX_calculate(H):
    D = len(H)
    n = int(math.log(len(H), 2))

    # diagonalize the Hamiltonian H
    l, V = np.linalg.eig(H)
    V_dag = V.conj().T

    # X matrix is (2^n, 2^n) , shrinked version of the original matrix
    # X_{j,i} = OriginalX_{ij,ij}
    X_mat = np.zeros((D, D), dtype=complex)
    for i in range(D):
        for j in range(D):
            for b in range(D):
                X_mat[j, i] += V_dag[i, b]*V_dag[j, b]*V[b, i]*V[b, j]
    X_inv = np.linalg.inv(X_mat)
    return l, V, X_mat, X_inv

def old_Direct_calculate_X(V):
    dim = len(V)
    V_dag = V.conj().T

    # X matrix is (2^n, 2^n) , shrinked version of the original matrix
    # X_{j,i} = OriginalX_{ij,ij}
    X_mat = np.zeros((dim, dim), dtype=complex)
    for i in tqdm(range(dim),leave=False):
        for j in range(dim):
            for b in range(dim):
                X_mat[j, i] += V_dag[i, b]*V_dag[j, b]*V[b, i]*V[b, j]
    X_inv = np.linalg.inv(X_mat)
    return X_mat, X_inv

def Direct_calculate_X(V):
    dim = len(V)
    
    V_abs = np.zeros((dim,dim),dtype = float)
    for i in tqdm(range(dim),leave=False):
        for j in range(dim):
            V_abs[i,j] = np.abs(V[i,j])**2

    X_mat = np.dot(V_abs.T,V_abs)
    X_inv = np.linalg.inv(X_mat)
    return X_mat, X_inv

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


def Get_rho(n):
    U0 = random_unitary(2**n).to_matrix()
    rho0 = np.diag([1]+[0 for i in range(2**n-1)])
    rho = np.dot(U0, np.dot(rho0, U0.conj().T))
    return rho

# CHANGE: ways to put in H
def Get_lvx(h_group_num, theta_table, n_table):
    l_lst = -1
    V_lst = []
    X_mat_lst = []
    X_inv_lst = []

    n_num = len(n_table)
    n_tqdm = range(n_num)
    # n_tqdm = tqdm(range(n_num),leave=False)
    for n_idx in n_tqdm:
        n = n_table[n_idx]
        dim = 2**n

        V_lst.append([])
        X_mat_lst.append([])
        X_inv_lst.append([])

        t0 = time.time()
        h_tqdm = tqdm(range(h_group_num),leave=False)
        for P_idx in h_tqdm:
            P = random_hermitian(dim).to_matrix()

            V_lst[n_idx].append([])
            X_mat_lst[n_idx].append([])
            X_inv_lst[n_idx].append([])
        
            theta_tqdm = tqdm(theta_table,leave=False)
            for theta in theta_tqdm:
                V = expm(1j*theta*P)
                X_mat, X_inv = Direct_calculate_X(V)
                
                V_lst[n_idx][P_idx].append(V)
                X_mat_lst[n_idx][P_idx].append(X_mat)
                X_inv_lst[n_idx][P_idx].append(X_inv)

        t1 = time.time()
        print('n=',n,': finished (runtime:',t1-t0,')')
    
    return l_lst,V_lst,X_mat_lst,X_inv_lst


########################################################################################
####################                                            ########################
####################           Starting from here               ########################
####################                                            ########################
########################################################################################


### receiving args
args = parse()
# DM = args.DM
# M = args.M
# times = args.times
h_group_num = args.hnum
# tol = args.tol
name = args.name

# n_table = [2,3,4,5,6,7,8]
n_table = args.nlist
n_num = len(n_table)

# theta_table = [0.002,0.004,0.006,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
# theta_table = [0.002,0.01,0.05,0.1,0.2,0.3]
theta_table = args.theta
theta_num = len(theta_table)

# 测的observable: obs0(Fidelity)
# obs_type = 0

ens_table = ['Hamiltonian','Global','Local']
ens_num = len(ens_table)
# address for saved data
addr = "./store/record_"+ name +".npy"

################################# Preparation #################################

print('')
print('')
print('H_group_num=',h_group_num)
print('n will take:',n_table)
print('theta will take:',theta_table)

print('-- start preprocessing l,V,X_mat,X_inv')

### 准备V, X_mat, X_inv
# V_lst[n_idx][P_idx][theta_idx]
_,V_lst,X_mat_lst,X_inv_lst = Get_lvx(h_group_num,theta_table,n_table)

for n_idx in range(n_num):
    n = n_table[n_idx]

    V_l = V_lst[n_idx]
    X_mat_l = X_mat_lst[n_idx]
    X_inv_l = X_inv_lst[n_idx]

    np.save('./prep/V_l/n'+str(n)+'.npy',V_l)
    np.save('./prep/X_mat_l/n'+str(n)+'.npy',X_mat_l)
    np.save('./prep/X_inv_l/n'+str(n)+'.npy',X_inv_l)

### 准备rho
# rho_lst[n_idx]
# for n_idx in range(n_num):
#     n = n_table[n_idx]

#     rho = Get_rho(n)

#     np.save('./prep/rho/n'+str(n)+'.npy',rho)
