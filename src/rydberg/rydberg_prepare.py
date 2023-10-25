import time, math, cmath, argparse, os
import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford
from qiskit.opflow import I, X, Y, Z
from qiskit.quantum_info import Pauli
from tqdm import tqdm

# Parameters

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-ntable', type=int, nargs='+', default = [3,4,5,6,7,8], help='qubit number table')
    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')
    parser.add_argument('-xdisttable', type=float, nargs = '+', default = [9,9.5,10], help = 'table: atom distance for shadow Hamiltonian')
    parser.add_argument('-randamptable', type=float, nargs = '+', default= [0,0.5,1], help = 'table: amplitude of the random factor in atom distance')

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

# CHANGE: ways to put in H
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


### receiving args
args = parse()
C0_ratio = args.Cratio
h_group_num = args.hnum

x_dist_table = args.xdisttable
x_dist_num = len(x_dist_table)
rand_amp_table = args.randamptable
rand_amp_num = len(rand_amp_table)

n_table = args.ntable
n_num = len(n_table)

print('')
print('')
print('H_group_num=',h_group_num)
print('n takes:',n_table)
print('atom distance table=',x_dist_table)
print('random amplitude table=',rand_amp_table)
print('C-C0 ratio=',C0_ratio)
print('-------- start preprocessing l,V,X_mat,X_inv')
print('')

for x_dist_idx in range(x_dist_num):
    x_dist = x_dist_table[x_dist_idx]

    for rand_amp_idx in range(rand_amp_num):
        rand_amp = rand_amp_table[rand_amp_idx]

        if x_dist == int(x_dist):
            x_dist = int(x_dist)
        if rand_amp == int(rand_amp):
            rand_amp = int(rand_amp)
        if C0_ratio == int(C0_ratio):
            C0_ratio = int(C0_ratio)

        # create folder
        suf_lst = ['l_l','V_l','X_mat_l','X_inv_l']
        for y in suf_lst:
            path = './shadow_LVX/xdist'+str(x_dist)+'_rand'+str(rand_amp)+'_C'+str(C0_ratio)+ '/' + y
            create(path)

        for n_idx in range(n_num):
            n = n_table[n_idx]

            l_lst,V_lst,X_mat_lst,X_inv_lst,H_lst = Get_lvx(n, h_group_num, C0_ratio, x_dist, rand_amp)

            prefix = './shadow_LVX/xdist'+str(x_dist)+'_rand'+str(rand_amp)+'_C'+str(C0_ratio)
            np.save(prefix + '/l_l/n'+str(n)+'.npy', l_lst)
            np.save(prefix + '/V_l/n'+str(n)+'.npy',V_lst)
            np.save(prefix + '/X_mat_l/n'+str(n)+'.npy',X_mat_lst)
            np.save(prefix + '/X_inv_l/n'+str(n)+'.npy',X_inv_lst)

        print('* xdist=', x_dist, ', rand=', rand_amp, ': finished preparation!')
        print('')

    
