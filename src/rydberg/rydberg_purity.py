import time,cmath,math,argparse,random, os

import numpy as np
import scipy as sp
from scipy.linalg import expm,sqrtm
import qiskit as qs
from qiskit.quantum_info import random_unitary, random_clifford, random_hermitian
from qiskit.quantum_info import Pauli
from tqdm import tqdm

import qiskit
I = qiskit.quantum_info.Pauli('I')
X = qiskit.quantum_info.Pauli('X')
Y = qiskit.quantum_info.Pauli('Y')
Z = qiskit.quantum_info.Pauli('Z')

# Parameters
global tol, evol_xdist
global data,ideal,snapshot
global t_idx,H_idx

def parse():
    parser = argparse.ArgumentParser(description='General framework of Hamiltonian shadow')
    parser.add_argument('-DM', type=int, default = 40000, help='the size of the entire data pool')
    parser.add_argument('-M', type=int, default = 10000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 1000, help='times of sampled subset from the datapool')
    parser.add_argument('-n', type=int, default = 12, help='the qubit number of the entire system')
    parser.add_argument('-hnum', type=int, default=10, help='size of the given Hamiltonians group')
    parser.add_argument('-htable', type=int, nargs = '+', default = [0,1,2,3,4,5,6,7,8,9], help='table of all the used Hamiltonians in the group')
    parser.add_argument('-t',type = float, nargs='+',default = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2] , help='list of time of all the smapshots')
    parser.add_argument('-tmax', type=int, default=10000, help='maximal time of twirling')
    parser.add_argument('-rho', type=int,default=1, help='type of initial state')

    parser.add_argument('-tol', type=int, default = -20, help='tolerance for Monte Carlo sampling')
    parser.add_argument('-Cratio', type=float, default = 1, help='ratio of C to C0 in the Rydberg Hamiltonian')
    parser.add_argument('-evoldist', type=float, default = 11, help='atom distance for evolution Hamiltonian')
    parser.add_argument('-xdist', type=float, default=9, help='atom distance for twirling Hamiltonian')
    parser.add_argument('-randamp', type=float, default=1, help='random factor in atom distance')

    parser.add_argument('-name', type=str, default='rydberg_purity', help = 'name of the data recorded')

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

def Shadow_estimator(rho, DM, l, V, X_mat, X_inv, t_min, t_max):

    DM_tqdm = tqdm(range(DM),leave=False)
    # DM_tqdm = range(DM)
        
    for i in DM_tqdm:
        rho_hat = Hamiltonian_Shadow(rho, l, V, X_mat, X_inv, t_min, t_max)
        snapshot[t_idx][H_idx][i] = rho_hat
    return

# calculate average_bias and variance
# return 2 variable, 1st: average bias; 2nd: log(variance)
# Take average over ${times} estimators
def performance(rho, DM, M, times, l, V, X_mat, X_inv, t_min, t_max):

    Shadow_estimator(rho, DM, l, V, X_mat, X_inv, t_min, t_max)

    half_DM = int(DM / 2)
    pool1 = snapshot[t_idx][H_idx][0:half_DM]
    pool2 = snapshot[t_idx][H_idx][half_DM:DM]
    tr_lst = [0]*times
    
    times_tqdm = tqdm(range(times),leave=False)
    # times_tqdm = range(times)
    for times_idx in times_tqdm:
        subset1 = random.sample(pool1,M)
        subset2 = random.sample(pool2,M)
        rho_mean1 = sum(subset1) / M
        rho_mean2 = sum(subset2) / M

        tr_lst[times_idx] = (np.trace(np.dot(rho_mean1,rho_mean2))).real

    tr_var = 0
    tr_mean = sum(tr_lst) / times
    for times_idx in range(times):
        tr_var += ((tr_lst[times_idx]-tr_mean)**2) / (times - 1)
    tr_dev = np.sqrt(tr_var)

    data[t_idx][H_idx][0] = tr_mean
    data[t_idx][H_idx][1] = tr_dev

    return tr_mean, tr_dev


# get an evoluted subsystem
def Get_rho(n,t,rho_type):
    # dim为子系统的维度，总维度为dim^2
    dim = 2**n
    n_total = 2*n
    # get a Hamiltonian without randomness
    C0 = 2*np.pi*(10)**6
    C = C0
    
    # xy_lst = [[evol_xdist*i, 0] for i in range(n_total)]
    xy_lst = [0]*n_total
    for i in range(n):
        xy_lst[i] = [evol_xdist*i, 0]
        xy_lst[n+i] = [evol_xdist*i, evol_xdist]

    omega_lst = [1.1*2*np.pi]*n_total
    delta_lst = [1.2*2*np.pi]*n_total
    phi_lst = [2.1]*n_total
    H = Rydberg_Hamiltonian(omega_lst, phi_lst, delta_lst, V_matrix(C, xy_lst))
    # H = random_hermitian
    
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

        xy_lst = [0]*n
        for i in range(n):
            xy_lst[i] = [x_dist*i+rand_amp*np.random.rand(), 0]
        # x_lst = [x_dist*i+rand_amp*np.random.rand() for i in range(n)]
        # omega_lst = [1.1*2*np.pi + 0.5*np.random.rand()]*n
        omega_lst = [1.1*2*np.pi]*n
        delta_lst = [1.2*2*np.pi]*n
        phi_lst = [2.1]*n

        H = Rydberg_Hamiltonian(omega_lst, phi_lst, delta_lst, V_matrix(C, xy_lst))
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

h_table = args.htable
h_real_num = len(h_table)
h_group_num = args.hnum

tol = args.tol

# Hamiltonian shadow 的演化时间
twirling_t_min = 2
twirling_t_max = args.tmax

DM = args.DM
M = args.M
times = args.times
n = int(args.n / 2) # 此处n为一半系统的qubit number
evol_xdist = args.evoldist

C0_ratio = args.Cratio
x_dist = args.xdist
rand_amp = args.randamp
rho_type=args.rho

t_table = args.t

t_num = len(t_table)

# 默认ens=Hamiltonian shadow
# 默认obs=purity

# address for saved data
addr_data = "./store/data_"+args.name+".npy"
# addr_ideal = "./store/ideal_"+args.name+".npy"
create("./store/")

# snapshot[t_idx][H_idx][DM_idx]
# 注意在这个程序里，snapshot始终是列表！
# data只存处理之后的数据，每一组实验存一个mean value存一个standard deviation
# 0处存mean value，1处存standard deviation

snapshot =[[[0 for _ in range(DM)] for _ in range(h_group_num)] for _ in range(t_num)]
data = np.zeros((t_num,h_group_num,2))
ideal = np.zeros((t_num))
trace_ideal = np.zeros((t_num))

print('')
print('M=', M, ', times=', times, ', H_group_num=', h_group_num)
print('The hamiltonian that we will run:', h_table )
print('Data pool size=', DM)
print('time list:',t_table)
print('qubit number of the entire system=',2*n)
print('atom distance of evolution Hamiltonian:', evol_xdist)
print('atom distance of shadow twirling Hamiltonian:', x_dist)
print('random factor in twirling Hamiltonian:', rand_amp)
print('rho_type=',rho_type)
print('twirling time: [', twirling_t_min, ',', twirling_t_max, ']')
print('**** Purity test:')

print('-- start preprocessing l,V,X_mat,X_inv')

# Get Hamiltonian H and then calculate l,V,X,X_inv, 改为从prep中获得VX
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

# prepare state rho
print('-- start preparing all states')
rho_lst = [0]*t_num
for t_idx in tqdm(range(t_num),leave=False):
    t = t_table[t_idx]
    rho = Get_rho(n,t,rho_type)
    rho_lst[t_idx] = rho

######################################## start RUNNING!

# prepare rho
rho_lst = [0]*t_num
for t_idx in tqdm(range(t_num),leave=False):
    t = t_table[t_idx]
    rho_lst[t_idx] = Get_rho(n,t,rho_type)

# calculate out the ideal value
print('------------ Theoretical entropy value:')
for t_idx in range(t_num):
    t = t_table[t_idx]

    rho = rho_lst[t_idx]
    trace_value = np.trace(np.dot(rho, rho)).real
    # real_value = - np.log2(trace_value)
    ideal[t_idx] = trace_value
    # trace_ideal[t_idx] = trace_value
    print('** t=', t, ': trace=', trace_value)
print('')

# start simulation
print('------------------------------- Start simulation:')
print('')

for H_idx in h_table:
    print('--------')
    print('**** Hamiltonian No.',H_idx,', subsystem n=', n, ':')
    print('')

    for t_idx in range(t_num):
        t = t_table[t_idx]

        # Get state rho
        rho = rho_lst[t_idx]

        t0 = time.time()
        res,dev = performance(
            rho, DM, M, times, l_lst[H_idx], V_lst[H_idx], X_mat_lst[H_idx], X_inv_lst[H_idx], twirling_t_min, twirling_t_max)
        t1 = time.time()
        # bias = - np.log2(res) - ideal[t_idx]
        # trace_bias = res - trace_ideal[t_idx]
        bias = res - ideal[t_idx]

        print('** t=', t,': purity bias=', bias, ', standard deviation=', dev, ' (runtime=', t1-t0, ')')
        print('')

    np.save(addr_data, data)
    # np.save(addr_ideal, ideal)