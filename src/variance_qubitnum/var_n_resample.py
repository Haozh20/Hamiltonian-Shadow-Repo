import numpy as np
import random,time,argparse
from tqdm import tqdm

# Parameters
global tol,record,ideal
global ens_idx,n_idx,theta_idx,P_idx

def parse():
    parser = argparse.ArgumentParser(description='data resampling of rydberg fidelity and ZXZ cluster')

    parser.add_argument('-M', type=int, default = 1000, help='number of rounds for one estimator')
    parser.add_argument('-times', type=int, default = 1000, help='times of sampled subset from the datapool')

    parser.add_argument('-name', type=str, default='Var_n_Fidelity', help = 'name of the data recorded')

    return parser.parse_args()

# including taking median over all P matrices
def resample(pool, M, times, h_group_num):
    
    # list that store each estimator, to numerically calculate variance
    est_lst = np.zeros((h_group_num,times))
    est_mean_lst = [0]*h_group_num
    est_var_lst = [0]*h_group_num

    # start running value of each Hamiltonians
    for P_idx in tqdm(range(h_group_num),leave=False):

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



########################################################################################
####################                                            ########################
####################           Starting from here               ########################
####################                                            ########################
########################################################################################


### receiving args
args = parse()

M = args.M
times = args.times
name = args.name

################################# Preparation #################################

addr_record = '../store/var_n/record_'+name+'.npy'
addr_data = '../store/var_n/data_'+name+'.npy'
# record[ens_idx][n_idx][theta_idx][P_idx][DM_idx]
record = np.load(addr_record)

# CHECK
r_shape = record.shape
print('record shape=', r_shape)
ens_num = r_shape[0]
n_num = r_shape[1]
theta_num = r_shape[2]
h_group_num = r_shape[3]
DM = r_shape[4]

# assumed parameter
ens_table = ['Hamiltonian','Global','Local']
n_table = [2,3,4,5,6,7,8,9,10]
theta_table = [0.1,0.2,0.3] 

# data[ens_idx][theta_idx][n_idx], only store the variance
data = np.zeros((ens_num, theta_num, n_num))


print('')
print('Original record: DM=', DM, ', ensemble number =', ens_num, ', n number =', n_num, ', theta number =', theta_num, ', h_group_num=', h_group_num)
print('Sample parameter: M=', M, ', times=', times, ', H_group_num=',h_group_num)
print('Experiment name=', name)
print('Start resampling:')
print('--------------------')

################################# Resampling Local shadow #################################

ens_idx = 2
theta_idx = 0
P_idx = 0
P_num = 1
print('')
print('-------------- Simulated Local shadow --------------')

for n_idx in range(n_num):
    n = n_table[n_idx]

    t0 = time.time()

    pool = [0]*P_num
    # OPTIMIZE: 所有P用同一组sampled index
    for P_idx in range(P_num):
        pool[P_idx] = list(record[ens_idx][n_idx][theta_idx][P_idx])

    # resample
    est_mean, est_var = resample(pool, M, times, P_num)
    
    t1 = time.time()

    data[ens_idx][theta_idx][n_idx] = est_var
    real_value = 1

    print('* (Local) n=', n, ': bias=', est_mean-real_value,', variance=', est_var, '(runtime:',t1-t0,')')

print('')
np.save(addr_data, data)

################################# Simulated Hamiltonian shadow #################################

ens_idx = 0
P_num = h_group_num
print('')
print('-------------- Resampling Hamiltonian shadow --------------')

for theta_idx in range(theta_num):
    theta = theta_table[theta_idx]
    print('------------')
    print('*** theta=', theta)
    print('')

    for n_idx in range(n_num):
        n = n_table[n_idx]

        t0 = time.time()

        pool = [0]*P_num
        # OPTIMIZE: 所有P用同一组sampled index
        for P_idx in range(P_num):
            # # RERUN
            # theta_idx = 0
            pool[P_idx] = list(record[ens_idx][n_idx][theta_idx][P_idx])

        # resample
        est_mean, est_var = resample(pool, M, times, P_num)

        t1 = time.time()

        data[ens_idx][theta_idx][n_idx] = est_var
        real_value = 1

        print('* (Hamiltonian) n=', n, ': bias=', est_mean-real_value,', variance=', est_var, '(runtime:',t1-t0,')')
    
    print('')
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

    est_var = record[ens_idx][n_idx][theta_idx][P_idx][DM_idx]

    t1 = time.time()

    data[ens_idx][theta_idx][n_idx] = est_var
    
    print('* (Global) n=', n, ': variance=', est_var, '(runtime:',t1-t0,')')
    
print('')
np.save(addr_data, data)