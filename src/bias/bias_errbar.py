import time, math, cmath, os
import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford, random_hermitian
from qiskit.opflow import I, X, Y, Z
from qiskit.quantum_info import Pauli
from tqdm import tqdm
import argparse

# Parameters
global tol,record,ideal
global ens_idx,obs_idx,n_idx,alpha_idx,H_idx

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-M', type=int, default = 10000000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 1, help='number of estimators')
    parser.add_argument('-nmin', type=int, default = 4, help='qubit number lower bound')
    parser.add_argument('-nmax', type=int, default = 4, help='squbit number upper bound')
    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')

    parser.add_argument('-obs', type=int, nargs='+', default=[0,4], help = 'index of all the observables')
    parser.add_argument('-alpha', type=int, nargs='+', default=[0,10,1,3], help='list of all the alpha to run')
    parser.add_argument('-name', type=str, default='bias_fidelity', help = 'name of the data recorded')
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

# ensemble 3: correct local Pauli shadow protocol
# TODO
def Local_Shadow(rho):
    return rho

# ensemble 4: Hamiltonian without shadow, use precise inverse map
# TODO
def Hamiltonian_precise(rho, l, V, X_mat, X_inv, t_min, t_max):
    return rho


def Shadow_estimator(rho, obs, obs_type, M, ensemble, l, V, X_mat, X_inv, t_min, t_max):
    res = 0
    
    # CHANGE: whether to show progress bar for M
    M_tqdm = tqdm(range(M),leave=False)
    # M_tqdm = range(M)
    if (obs_type==0) or (obs_type==1) or (obs_type==4): # Fidelity / Pauli Operator / Entanglement Witness

        real_value = np.trace(np.dot(obs, rho))
        ideal[ens_idx][obs_idx][n_idx][alpha_idx][H_idx] = real_value.real
        
        if ensemble == 0:
            for i in M_tqdm:
                rho_hat = Wrong_Global_Shadow(rho, l, V, t_min, t_max)
                term = np.trace(np.dot(obs, rho_hat))
                res += term/M
                record[ens_idx][obs_idx][n_idx][alpha_idx][H_idx][i] = term.real
        elif ensemble == 1:
            for i in M_tqdm:
                rho_hat = Hamiltonian_Shadow(rho, l, V, X_mat, X_inv, t_min, t_max)
                term = np.trace(np.dot(obs, rho_hat))
                res += term/M
                record[ens_idx][obs_idx][n_idx][alpha_idx][H_idx][i] = term.real
        elif ensemble == 2:
            for i in M_tqdm:
                rho_hat = Original_Global_Shadow(rho)
                term = np.trace(np.dot(obs, rho_hat))
                res += term/M
                record[ens_idx][obs_idx][n_idx][alpha_idx][H_idx][i] = term.real

    # # Trace distance
    # elif obs_type==2:
    #     if ensemble == 0:
    #         for i in M_tqdm:
    #             res += Wrong_Global_Shadow(rho, l, V, t_min, t_max)/M
    #     elif ensemble == 1:
    #         for i in M_tqdm:
    #             res += Hamiltonian_Shadow(rho, l, V, X_mat, X_inv, t_min, t_max)/M
    #     elif ensemble == 2:
    #         for i in M_tqdm:
    #             res += Original_Global_Shadow(rho)/M
    return res

# calculate average_bias and variance
# return 2 variable, 1st: average bias; 2nd: log(variance)
# Take average over ${times} estimators
def performance(rho, obs, obs_type, M, times, ensemble, l, V, X_mat, X_inv, t_min, t_max):
    var = 0
    bias = 0
    abs_bias = 0

    # times_tqdm = tqdm(range(times),leave=False)
    # times_tqdm = range(times)
    # for i in times_tqdm:
    estimator = Shadow_estimator(rho, obs, obs_type, M, ensemble, l, V, X_mat, X_inv, t_min, t_max)
    
    if obs_type==0: # Fidelity
        real_value = 1
        term = (estimator - real_value).real
    elif (obs_type==1) or (obs_type==4): # Pauli Operator / Entanglement Witness
        real_value = np.trace(np.dot(obs, rho))
        term = (estimator - real_value).real
    
    elif obs_type==2: # Trace distance
        term = (0.5*np.trace(sqrtm(np.dot((rho-estimator).conj().T,rho-estimator)))).real

    var += (term**2)/times
    bias += term/times
    abs_bias += np.abs(term)/times

    return bias, abs_bias, var

# CHANGE: ways to put in H
def Get_lvx(h_group_num, alpha_table, n):
    l_lst = []
    V_lst = []
    X_mat_lst = []
    X_inv_lst = []

    for i in range(h_group_num):
        H0 = random_hermitian(2**n).data
        L = np.diag([np.random.rand() for i in range(2**n)])

        V_lst.append([])
        X_mat_lst.append([])
        X_inv_lst.append([])
        l_lst.append([])
     
        for alpha_idx in range(len(alpha_table)):
            alpha = alpha_table[alpha_idx]
            H = H0 + alpha*L
            l, V, X_mat, X_inv = VX_calculate(H)
            V_lst[i].append(V)
            X_mat_lst[i].append(X_mat)
            X_inv_lst[i].append(X_inv)
            l_lst[i].append(l)
    
    return l_lst,V_lst,X_mat_lst,X_inv_lst

def Get_rho(n):
    dim = 2**n
    # random pure state
    # U0 = random_unitary(dim).to_matrix()
    # rho0 = np.diag([1]+[0 for i in range(dim-1)])
    # rho = np.dot(U0, np.dot(rho0, U0.conj().T))

    # GHZ state
    rho = np.zeros((dim,dim))
    rho[0,0] = 0.5
    rho[0,dim-1] = 0.5
    rho[dim-1,0] = 0.5
    rho[dim-1,dim-1] = 0.5

    return rho

# CHANGE: need changing when measure Hamiltonian
def Get_obs(obs_type,rho):
    n = int(math.log(len(rho), 2))
    dim = len(rho)
    
    if obs_type==0: # Fidelity
        obs = rho
     
    elif obs_type==1: # Pauli
        z = [0]*n
        x = [1]*n
        obs = Pauli((z,x,0)).to_matrix()
    
    elif obs_type==2: # trace distance
        obs=np.zeros((dim,dim))
    
    elif obs_type==4: # Entanglement Witness
        if n==4:
            obs = np.zeros((dim,dim))
            obs[3,3] = 0.5
            obs[15,0] = -0.5
            obs[0,15] = -0.5
            obs[12,12] = 0.5
        else:
            print('ERROR: qubit number n is not 4, neet to rewrite!!!')

    return obs

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
M = args.M
times = args.times
n_min = args.nmin
n_max = args.nmax
C0_ratio = args.Cratio
h_group_num = args.hnum
tol = args.tol

t_max = 10000
t_min = 2
M_table = [1000,5000,10000,50000,100000,500000,1000000,5000000]

# 测的observable: obs0(Fidelity/Entanglement Witness)
obs_table = args.obs
obs_num = len(obs_table)
# adder list
alpha_table = args.alpha
alpha_num = len(alpha_table)
# record[0]对应ens1(Hamiltonian), record[1]对应ens2(Original Global)
ens_table = [0,1]
ens_num = len(ens_table)
# address for saved data
addr_record = "./store/record_"+args.name+".npy"
addr_ideal = "./store/ideal_"+args.name+".npy"

create("./store/")

# CHANGE: arrangement of record
# record[ens_idx][obs_idx][n_idx][alpha_idx][H_idx][M]
# ens_table: [Wrong global, Hamiltonian]
# obs_table: [Fidelity] or [Pauli]
record = np.zeros((ens_num,obs_num,n_max-n_min+1,alpha_num,h_group_num,M))
ideal = np.zeros((ens_num,obs_num,n_max-n_min+1,alpha_num,h_group_num))

for obs_idx in range(len(obs_table)):
    obs_type = obs_table[obs_idx]
    print('')
    print('')
    print('H_group_num=',h_group_num)
    print('t in [', t_min, ', ', t_max, ']')
    print('M=', M, ', the list of K: ', M_table)
    print('The list of alpha: ', alpha_table)
    print('The list of observable: ', obs_table)
    print('The list of ensemble protocol ', ens_table)

    print('--------------------------------')
    if obs_type==0:
        print('**** Fidelity test:')
    elif obs_type==1:
        print('**** Pauli test:')
    elif obs_type==2:
        print('**** trace distance test:')
    elif obs_type==3:
        print('**** Hamiltonian observable test:')
    elif obs_type==4:
        print('**** Entanglement Witness test:')

    for n in range(n_min, n_max+1):
        n_idx = n-n_min
        print('--------------')
        print('n=',n)
        print('')

        # Get state rho
        rho = Get_rho(n)

        # Get observable obs
        obs = Get_obs(obs_type,rho)

        # Get Hamiltonian H and then calculate l,V,X,X_inv
        l_lst,V_lst,X_mat_lst,X_inv_lst = Get_lvx(h_group_num, alpha_table, n)

        # CHANGE: different loop and different index
        # select different coefficient alpha
        for alpha_idx in range(len(alpha_table)):
            alpha = alpha_table[alpha_idx]
            print('-----------')
            print('**** alpha=',alpha)
            print('')

            # select different M for different experiments
            # for M_idx in range(len(M_table)):
            #     M = M_table[M_idx]
            #     print('----')
            #     print('** M=',M)
            #     print('')

            res = 0
            # h_tqdm = tqdm(range(h_group_num),leave=False)
            h_tqdm = range(h_group_num)
            for H_idx in h_tqdm:
                for ens_idx in range(ens_num):
                    ens_type = ens_table[ens_idx]

                    t0 = time.time()
                    bias, abs_bias, var = performance(
                        rho, obs, obs_type, M, times, ens_type, l_lst[H_idx][alpha_idx], V_lst[H_idx][alpha_idx], X_mat_lst[H_idx][alpha_idx], X_inv_lst[H_idx][alpha_idx], t_min, t_max)
                    # CHANGE: choose one to show performance
                    res = bias
                    t1 = time.time()

                    # CHANGE: messages to print
                    if ens_type==0: # Wrong global shadow
                        print('* Wrong Global shadow, H(', H_idx,'): total mean of average bias=', res, ' (runtime=', t1-t0, ')')
                    elif ens_type==1: # Hamiltonian shadow
                        print('* Hamiltonian shadow, H(', H_idx,'): total mean of average bias=', res, ' (runtime=', t1-t0, ')')
                    elif ens_type==2: # Original Global shadow
                        print('* Original Global shadow, H(', H_idx,'): total mean of average bias=', res, ' (runtime=', t1-t0, ')')
              
                print('')
                
        np.save(addr_record, record)
        np.save(addr_ideal, ideal)