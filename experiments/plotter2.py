# Adapted from Vinjai's vinjai_plots.py
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# deets format:
# (this scheme is all messed up and not consistent with early, middle, or late runs)
# First digit: Adversary's comm. 0: no noise, 2: doesn't get comm, x in [3, ..., 10]: noise amount x/10
# Second digit: comm protocol.  0: encrypted; 1: not encrypted (later will be size of key)
# Third digit: 3 is not sqrt, 4 is sqrt
# Fourth digit: 0 normally, 1 is modded env, 2 is modded env + modded rewards
# Fifth digit: now 3-5+: size of world.dim_c
# BIG: Trained with BIG MLP (5 layers, num_units = 128)

def open_pickle(deets):

    with open('./learning_curves/adversary_simple_listener'+deets+'_agrewards.pkl', 'rb') as f:
        data = pickle.load(f)
    return np.array(data)

#speaker-listener rewards
def get_sl_rewards(deets):
    all_rewards = open_pickle(deets)
    rewards = all_rewards[:,0]
    return rewards

def get_adversary_rewards(deets):
    all_rewards = open_pickle(deets)
    rewards = all_rewards[:,2]
    return rewards

def get_listener_distance(deets):
    return -(get_sl_rewards(deets) + get_adversary_rewards(deets))



def var_comparison(deets_list, labels=None, title="", which=0):
    if (which==0):
        rewards_func = get_sl_rewards
        who = 'speaker/listener'
    if (which==1):
        rewards_func = get_adversary_rewards
        who = 'adversary'
    if (which==2):
        rewards_func = get_listener_distance
        who = "listener"

    rewards_list = [rewards_func(deets) for deets in deets_list]

    if labels is None:
        labels = deets_list

    print(rewards_list)   
    x = np.arange(0,120)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(rewards_list)):
        ax.plot(x, rewards_list[i], label=labels[i])
    ax.legend()
    ax.set_xlabel('# 1000s of episodes')
    
    if which==2:
        ax.set_ylabel(who + ' distance')
    else:
        ax.set_ylabel(who + ' reward')

    if title == "":
        ax.set_title('Influence of comms on '+who+' performance')
    else:
        ax.set_title(title)
    plt.ylim([-10,30])
    plt.show()

normalcomm = ['-VAR-TEST-1', '-VAR-TEST-2', '-VAR-TEST-3', '-VAR-TEST-4','-VAR-TEST-5', 
            '-VAR-TEST-6','-VAR-TEST-7', '-VAR-TEST-8', '-VAR-TEST-9', '-VAR-TEST-10']

normal = ['-STD-TEST-1', '-STD-TEST-2', '-STD-TEST-3', '-STD-TEST-4','-STD-TEST-5', 
            '-STD-TEST-6','-STD-TEST-7', '-STD-TEST-8', '-STD-TEST-9', '-STD-TEST-10',
            '-STD-TEST-11', '-STD-TEST-12', '-STD-TEST-13', '-STD-TEST-14','-STD-TEST-15', 
            '-STD-TEST-16','-STD-TEST-17', '-STD-TEST-18','-STD-TEST-19','-STD-TEST-20',
            '-STD-TEST-21','-STD-TEST-22','-STD-TEST-23', '-STD-TEST-24','-STD-TEST-25',
            '-STD-TEST-27','-STD-TEST-28','-STD-TEST-29','-STD-TEST-30'
            ]

corners = ['_alt-STD-TEST-1', '_alt-STD-TEST-2', '_alt-STD-TEST-3', '_alt-STD-TEST-4','_alt-STD-TEST-5', 
            '_alt-STD-TEST-6','_alt-STD-TEST-7', '_alt-STD-TEST-8', '_alt-STD-TEST-9', '_alt-STD-TEST-10',
            '_alt-STD-TEST-12', '_alt-STD-TEST-13', '_alt-STD-TEST-14','_alt-STD-TEST-15', 
            '_alt-STD-TEST-16','_alt-STD-TEST-17', '_alt-STD-TEST-18','_alt-STD-TEST-19','_alt-STD-TEST-20',
            '_alt-STD-TEST-21','_alt-STD-TEST-22','_alt-STD-TEST-23', '_alt-STD-TEST-24','_alt-STD-TEST-25',
            '_alt-STD-TEST-26','_alt-STD-TEST-27','_alt-STD-TEST-28'
            ]
#corners = ['_alt-STD-TEST-1', '_alt-STD-TEST-2', '_alt-STD-TEST-3', '_alt-STD-TEST-4','_alt-STD-TEST-5', 
#            '_alt-STD-TEST-6','_alt-STD-TEST-7', '_alt-STD-TEST-8', '_alt-STD-TEST-9', '_alt-STD-TEST-10']

cornerinfo = ['_alt2-VAR-TEST-1', '_alt2-VAR-TEST-2', '_alt2-VAR-TEST-3', '_alt2-VAR-TEST-4','_alt2-VAR-TEST-5', 
            '_alt2-VAR-TEST-6','_alt2-VAR-TEST-7', '_alt2-VAR-TEST-8', '_alt2-VAR-TEST-9', '_alt2-VAR-TEST-10']


var_comparison(corners, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'], 
    "Speaker/Listener Performance when Adversary Recieves Noise", which=0)

def fill_comparison(deets_list, labels=None, title="", which=0):
    if (which==0):
        rewards_func = get_sl_rewards
        who = 'speaker/listener'
    if (which==1):
        rewards_func = get_adversary_rewards
        who = 'adversary'
    if (which==2):
        rewards_func = get_listener_distance
        who = "listener"

    rewards_list = [rewards_func(deets) for deets in deets_list]

    r_np = np.asarray(rewards_list)
    r_mean = np.mean(rewards_list, axis=0)
    r_std = np.std(rewards_list, axis=0)
    top = r_mean + 2*r_std
    bottom = r_mean - 2*r_std

    colorb = ['#f7c6c6', mcolors.cnames['darkred']]

    if labels is None:
        labels = deets_list

    print(rewards_list)   
    x = np.arange(0,120)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, r_mean, color=colorb[1])
    ax.fill_between(x, bottom, top, color=colorb[1], alpha = 0.2)

    ax.legend()
    ax.set_xlabel('# 1000s of episodes')
    
    if which==2:
        ax.set_ylabel(who + ' distance')
    else:
        ax.set_ylabel(who + ' reward')

    if title == "":
        ax.set_title('Influence of comms on '+who+' performance')
    else:
        ax.set_title(title)
    plt.ylim([-10,30])
    plt.show()

#fill_comparison(normalcomm, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
#            "Speaker/Listener Performance when Adversary Recieves Comm", which=0)
