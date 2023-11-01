import time,math,cmath,argparse,random,os
import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford, random_hermitian
from qiskit.opflow import I, X, Y, Z
from qiskit.quantum_info import Pauli
from tqdm import tqdm

# Parameters
global tol,record,data
global ens_idx,n_idx,theta_idx,P_idx

def parse():
    parser = argparse.ArgumentParser(description='relationship of variance and qubit number')
    parser.add_argument('-DM', type=int, default = 5000, help='the size of the entire data pool')
    parser.add_argument('-M', type=int, default = 1000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 1000, help='times of sampled subset from the datapool')
    parser.add_argument('-hnum', type=int, default = 10, help='number of Hamiltonians in the group')
    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    
    parser.add_argument('-theta',type = float, nargs='+',default=[0.2,0.3], help='list of all the theta')
    parser.add_argument('-nlist',type = int, nargs='+',default=[2,3,4,5,6,7,8,9,10], help='list of all the n')

    parser.add_argument('-name', type=str, default='var_qubitnum_relation_fidelity', help = 'name of the data recorded')
    return parser.parse_args()

def Direct_calculate_X(V):
    dim = len(V)
    V_dag = V.conj().T

    # X matrix is (2^n, 2^n) , shrinked version of the original matrix
    # X_{j,i} = OriginalX_{ij,ij}
    X_mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for b in range(dim):
                X_mat[j, i] += V_dag[i, b]*V_dag[j, b]*V[b, i]*V[b, j]
    X_inv = np.linalg.inv(X_mat)
    return X_mat, X_inv

def digit_represent(num,k,n):
    res = [0]*n
    for i in range(n):
        t = num % k
        num = int((num - t) / k)
        res[n-1-i] = t
    return res

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


# Variance of global shadow
def Var_global(obs,rho,M):
    dim = len(rho)
    part1 = np.trace(np.dot(obs,obs)) + 2*np.trace(np.dot(np.dot(obs,obs),rho))
    part2 = 2*np.trace(obs)*np.trace(np.dot(obs,rho)) + (np.trace(obs))**2
    part3 = (np.trace(np.dot(obs,rho)))**2
    var = ((dim+1)*part1)/(dim+2) - part2/(dim+2) - part3
    var = var.real / M
    return var

# ensemble 0: Our shadow using Hamiltonian
def Hamiltonian_Shadow(rho, V, X_mat, X_inv):
    dim = len(rho)
    L = expm(2j*np.pi*np.diag([np.random.rand() for i in range(dim)]))
    U = np.dot(V,np.dot(L,V.conj().T))
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


# ensemble 2: Local shadow
def Local_Shadow(rho):
    dim = len(rho)
    n = int(math.log(dim,2))

    # independently sample clifford gate on each qubit
    # U, the big Unitary on n qubits
    u0 = random_clifford(1).to_matrix()
    u_lst = [u0]
    U = u0
    for i in range(n-1):
        u = random_clifford(1).to_matrix()
        U = np.kron(U,u)
        u_lst.append(u)
    
    evoluted_rho = np.dot(U,np.dot(rho,U.conj().T))
    prob_lst = [0 for i in range(dim)]
    for i in range(dim):
        if abs(evoluted_rho[i][i]) > 10**(tol):
            prob_lst[i] = evoluted_rho[i][i].real
        else: 
            prob_lst[i] = 0
    prob_lst = prob_lst / sum(prob_lst)
    measurement_res = np.random.choice(np.arange(0,dim),p = prob_lst)
    
    res_bin = digit_represent(measurement_res,2,n)
    rho_hat_lst = []
    Rho_hat = 1
    for i in range(n):
        b_i = res_bin[i]
        u_i = u_lst[i] 
        rho_hat_i = np.outer(u_i[b_i,:].conj(),u_i[b_i,:])
        rho_hat_i = 3*rho_hat_i - np.eye(2)
        rho_hat_lst.append(rho_hat_i)

        Rho_hat = np.kron(Rho_hat,rho_hat_i)
    return Rho_hat

def Shadow_estimator(rho, obs, obs_type, DM, ensemble, V, X_mat, X_inv):
    res = 0
    # Fidelity / Pauli Operator
    # CHANGE: whether to show progress bar for DM
    DM_tqdm = tqdm(range(DM),leave=False)
    if (obs_type==0) or (obs_type==1):
        if ensemble == 0: # Hamiltonian shadow
            for i in DM_tqdm:
                rho_hat = Hamiltonian_Shadow(rho, V, X_mat, X_inv)
                term = np.trace(np.dot(obs, rho_hat))
                record[ens_idx][n_idx][theta_idx][P_idx][i] = term.real

        elif ensemble == 2: # Local shadow
            for i in DM_tqdm:
                rho_hat = Local_Shadow(rho)
                term = np.trace(np.dot(obs, rho_hat))
                record[ens_idx][n_idx][theta_idx][P_idx][i] = term.real
    return

# including taking median over all P matrices
def resample(pool, M, times, median_num):
    
    # list that store each estimator, to numerically calculate variance
    est_lst = np.zeros((median_num,times))
    est_mean_lst = [0]*median_num
    est_var_lst = [0]*median_num

    # start running value of each Hamiltonians
    for P_idx in tqdm(range(median_num),leave=False):

        # Start sampling for <times> times
        for times_idx in range(times):

            subset = random.sample(pool[P_idx],M)
            mean_term = np.mean(subset)

            # get the median of all values
            est_lst[P_idx][times_idx] = mean_term
    
        # get the variance of the pool
        est_mean_lst[P_idx] = np.mean(est_lst[P_idx])
        est_var_lst[P_idx] = np.var(est_lst[P_idx]) * (times / (times-1))

    est_mean = np.median(est_mean_lst)
    est_var = np.median(est_var_lst)

    return est_mean, est_var


def performance_var(rho, obs, obs_type, DM, M, times, ensemble, V, X_mat, X_inv):
    total_var = 0

    # run all the experiments
    Shadow_estimator(rho, obs, obs_type, DM, ensemble, V, X_mat, X_inv)

    # define P_num
    if ens_idx==2: # local
        P_num = 1
    elif ens_idx==0: # hamiltonian
        P_num = 10
    
    pool = [0]*P_num
    for P_idx in range(P_num):
        pool[P_idx] = list(record[ens_idx][n_idx][theta_idx][P_idx])

    # resample
    est_mean, est_var = resample(pool, M, times, P_num)
    real_value = 1
    est_bias = est_mean - real_value

    # record into data
    data[ens_idx][theta_idx][n_idx] = est_var


    # pool = list(record[ens_idx][n_idx][theta_idx][P_idx])
    # t_sum = 0
    # sqr_sum = 0
    # times_tqdm = tqdm(range(times),leave=False)
    # # times_tqdm = range(times)
    # for times_idx in times_tqdm:
    #     # # DEBUG：相当于每次重新跑一遍
    #     # Shadow_estimator(rho, obs, obs_type, DM, ensemble, V, X_mat, X_inv)
    #     # pool = list(record[ens_idx][n_idx][theta_idx][P_idx])
    #     subset = random.sample(pool,M)
    #     term = sum(subset) / M
    #     # print(term)
    #     t_sum += term
    #     sqr_sum += term**2
    # total_var = (sqr_sum / (times - 1)) - (t_sum**2 / (times*(times - 1)))

    return est_bias, est_var

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


def Get_rho(n):
    # # 随机纯态
    # U0 = random_unitary(2**n).to_matrix()
    # rho0 = np.diag([1]+[0 for i in range(2**n-1)])
    # rho = np.dot(U0, np.dot(rho0, U0.conj().T))

    # GHZ态
    dim = 2**n
    rho = np.zeros((dim,dim))
    rho[0,0] = 0.5
    rho[0,dim-1] = 0.5
    rho[dim-1,0] = 0.5
    rho[dim-1,dim-1] = 0.5

    return rho

# CHANGE: need changing when measure Hamiltonian
def Get_obs(rho):   
    obs = rho
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
DM = args.DM
M = args.M
times = args.times
h_group_num = args.hnum
tol = args.tol
name = args.name

# n_table = [2,3,4,5,6,7,8]
n_table = args.nlist
n_num = len(n_table)

# theta_table = [0.002,0.004,0.006,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
# theta_table = [0.002,0.01,0.05,0.1,0.2,0.3]
# theta_table = [0.01,0.05,0.2]
theta_table = args.theta
theta_num = len(theta_table)

# 测的observable: obs0(Fidelity)
obs_type = 0

ens_table = ['Hamiltonian','Global','Local']
ens_num = len(ens_table)
# address for saved data
addr_data = "./store/data_"+ name +".npy"
create("./store/")

################################# Preparation #################################

print('')
print('')
print('M=', M, ', times=', times, ', H_group_num=',h_group_num)
print('Data pool size=',DM)
print('n will take:',n_table)
print('theta will take:',theta_table)


# record[ens_idx][n_idx][theta_idx][P_idx][DM_idx]
record = np.zeros((ens_num,n_num,theta_num,h_group_num,DM))

# data[ens_idx][theta_idx][n_idx], only store the variance
data = np.zeros((ens_num, theta_num, n_num))

# Get state rho
# rho_lst[n_idx]
rho_lst = [0]*n_num
for n_idx in range(n_num):
    n = n_table[n_idx]
    rho_lst[n_idx] = Get_rho(n_table[n_idx])
    # 改为从prep中读取rho
    # rho_lst[n_idx] = np.load('./prep/rho/n'+str(n)+'.npy')


# Get observable obs
# obs_lst[n_idx]
obs_lst = []
for n_idx in range(n_num):
    obs_lst.append(Get_obs(rho_lst[n_idx]))

# Get Hamiltonian H and then calculate l,V,X,X_inv
# V_lst[n_idx][P_idx][theta_idx]
print('-- start preprocessing l,V,X_mat,X_inv')

# _,V_lst,X_mat_lst,X_inv_lst = Get_lvx(h_group_num,theta_table,n_table)
# # 改为直接从prep里读取
# V_lst = [0]*n_num
# X_mat_lst = [0]*n_num
# X_inv_lst = [0]*n_num
# for n_idx in range(n_num):
#     n = n_table[n_idx]
#     V_lst[n_idx] = np.load('./prep/V_l/n'+str(n)+'.npy')
#     X_mat_lst[n_idx] = np.load('./prep/X_mat_l/n'+str(n)+'.npy')
#     X_inv_lst[n_idx] = np.load('./prep/X_inv_l/n'+str(n)+'.npy')

_,V_lst,X_mat_lst,X_inv_lst = Get_lvx(h_group_num,theta_table,n_table)

# prep进行适当调整
# V_lst = V_lst[6:7,:,:]
# X_mat_lst = X_mat_lst[6:7,:,:]
# X_inv_lst = X_inv_lst[6:7,:,:]


print('--------------------------------')
print('**** Fidelity test:')


################################# Simulated Local shadow #################################

ens_idx = 2
theta_idx = 0
P_idx = 0
P_num = 1
print('')
print('-------------- Local shadow --------------')

for n_idx in range(n_num):
    n = n_table[n_idx]

    t0 = time.time()

    # run all the experiments
    Shadow_estimator(rho_lst[n_idx], obs_lst[n_idx], obs_type, DM, ens_idx, V_lst[n_idx][P_idx][theta_idx], X_mat_lst[n_idx][P_idx][theta_idx], X_inv_lst[n_idx][P_idx][theta_idx])
    
    pool = [0]*P_num
    for P_idx in range(P_num):
        pool[P_idx] = list(record[ens_idx][n_idx][theta_idx][P_idx])

    # resample
    est_mean, est_var = resample(pool, M, times, P_num)
    real_value = 1
    est_bias = est_mean - real_value

    # record into data
    data[ens_idx][theta_idx][n_idx] = est_var

    # est_bias, est_var = performance_var(rho_lst[n_idx], obs_lst[n_idx], obs_type, DM, M, times, ens_idx, V_lst[n_idx][P_idx][theta_idx], X_mat_lst[n_idx][P_idx][theta_idx], X_inv_lst[n_idx][P_idx][theta_idx])
    
    t1 = time.time()

    print('* (Local)  n=', n, ': bias=', est_bias,', variance=', est_var, '(runtime:',t1-t0,')')

print('')
np.save(addr_data, data)


################################# Simulated Hamiltonian shadow #################################

ens_idx = 0
P_num = h_group_num
print('')
print('-------------- Hamiltonian shadow --------------')

for theta_idx in range(theta_num):
    theta = theta_table[theta_idx]
    print('---------')
    print('*** theta=', theta)
    print('')

    for n_idx in range(n_num):
        n = n_table[n_idx]

        

        t0 = time.time()

        # run all the experiments
        P_tqdm = tqdm(range(P_num),leave=False)
        for P_idx in P_tqdm:
            Shadow_estimator(rho_lst[n_idx], obs_lst[n_idx], obs_type, DM, ens_idx, V_lst[n_idx][P_idx][theta_idx], X_mat_lst[n_idx][P_idx][theta_idx], X_inv_lst[n_idx][P_idx][theta_idx])
        
        # creating pool
        pool = [0]*P_num
        for P_idx in range(P_num):
            pool[P_idx] = list(record[ens_idx][n_idx][theta_idx][P_idx])

        # resample
        est_mean, est_var = resample(pool, M, times, P_num)
        real_value = 1
        est_bias = est_mean - real_value

        # record into data
        data[ens_idx][theta_idx][n_idx] = est_var

        
        # for P_idx in P_tqdm:
        #     res = performance_var(rho_lst[n_idx], obs_lst[n_idx], obs_type, DM, M, times, ens_idx, V_lst[n_idx][P_idx][theta_idx], X_mat_lst[n_idx][P_idx][theta_idx], X_inv_lst[n_idx][P_idx][theta_idx])
        #     res_mean += res
        # res_mean = res_mean / h_group_num

        t1 = time.time()
        print('* (Hamiltonian)  n=', n, ': bias=', est_bias,', variance=', est_var, '(runtime:',t1-t0,')')
    
np.save(addr_data, data)

################################# Ideal Global shadow #################################

ens_idx = 1
theta_idx = 0
P_idx = 0
DM_idx = 0
print('')
print('-------------- Ideal Global shadow --------------')

for n_idx in range(n_num):
    n = n_table[n_idx]

    t0 = time.time()
    est_var = Var_global(obs_lst[n_idx],rho_lst[n_idx],M)
    record[ens_idx][n_idx][theta_idx][P_idx][DM_idx] = est_var
    data[ens_idx][theta_idx][n_idx] = est_var
    t1 = time.time()
    print('* (Global shadow)  n=', n, ': variance=', est_var, '(runtime:',t1-t0,')')
    print('')

np.save(addr_data, data)
