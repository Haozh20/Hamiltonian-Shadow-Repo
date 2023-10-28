import os
import numpy as np
import random,time,argparse
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description='data resampling of bias')
    parser.add_argument('-aidx', type=int, default= 1)
    parser.add_argument('-obsidx', type=int, default = 0)
    parser.add_argument('-Hidx', type=int, default= 0, help='the index of Hamiltonian corresponding to the experiment')
    parser.add_argument('-times', type=int, default=1000)

    return parser.parse_args()

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)

args = parse()
alpha_idx = args.aidx
obs_idx = args.obsidx
H_idx = args.Hidx
times = args.times

DM = 10000000

ens_table = [0,1]
ens_num = len(ens_table)
h_group_num = 10

M_table = [100,300,1000,3000,10000,30000,100000,300000,1000000]
M_num = len(M_table)

name_lst = [[0 for _ in range(2)] for _ in range(4)]
name_lst[0][0] = 'bias_errbar_GHZ_alpha0_fidelity'
name_lst[0][1] = 'bias_errbar_GHZ_alpha0_EW'
name_lst[1][0] = 'bias_errbar_GHZ_alpha1_fidelity'
name_lst[1][1] = 'bias_errbar_GHZ_alpha1_EW'
name_lst[2][0] = 'bias_errbar_GHZ_alpha3_fidelity'
name_lst[2][1] = 'bias_errbar_GHZ_alpha3_EW'
name_lst[3][0] = 'bias_errbar_GHZ_alpha10_fidelity'
name_lst[3][1] = 'bias_errbar_GHZ_alpha10_EW'

# data[alpha_idx][obs_idx][H_idx][ens_idx][M_idx][mean / std]
data = np.zeros((4,2,h_group_num,ens_num,M_num,2))


print('alpha_idx=',alpha_idx,', obs_idx=',obs_idx, 'H_idx=', H_idx)
name = name_lst[alpha_idx][obs_idx]
print('name=',name)
addr = './store/record_'+name+'.npy'
print('addr=',addr)

# record[ens_idx][0][0][0][H_idx][M]
record = np.load(addr)
print('shape of record:',record.shape)
print('')

pool0 = record[0][0][0][0][H_idx]
pool1 = record[1][0][0][0][H_idx]

if alpha_idx != 3:
    pool0 = pool0 * DM
    pool1 = pool1 * DM

pool0 = list(pool0)
pool1 = list(pool1)

# different choice of sample size

for M_idx in range(M_num):
    M = M_table[M_idx]

    est_lst0 = [0]*times
    est_lst1 = [0]*times

    # after fixing M, sample subset for <times> times
    t0 = time.time()
    for times_idx in tqdm(range(times),leave=False):
        subset0 = random.sample(pool0,M)
        subset1 = random.sample(pool1,M)
        est_lst0[times_idx] = np.mean(subset0)
        est_lst1[times_idx] = np.mean(subset1)
    est_mean0 = est_lst0[137]
    est_dev0 = np.std(est_lst0) *np.sqrt(times / (times-1))
    est_mean1 = est_lst1[137]
    est_dev1 = np.std(est_lst1) *np.sqrt(times / (times-1))

    data[alpha_idx][obs_idx][H_idx][0][M_idx][0] = est_mean0
    data[alpha_idx][obs_idx][H_idx][0][M_idx][1] = est_dev0
    data[alpha_idx][obs_idx][H_idx][1][M_idx][0] = est_mean1
    data[alpha_idx][obs_idx][H_idx][1][M_idx][1] = est_dev1

    t1 = time.time()

    print('**** M=',M, '  ( H_idx=', H_idx, ')  (runtime:',t1-t0,')')
    print('Wrong shadow: mean=',est_mean0, ', standard deviation=',est_dev0)
    print('Hamiltonian shadow: mean=',est_mean1, ', standard deviation=',est_dev1)
    print('')

create("'./data/median/")
np.save('./data/median/newdata_bias_H'+str(H_idx)+'_a'+str(alpha_idx)+'_obs'+str(obs_idx)+'.npy', data)


