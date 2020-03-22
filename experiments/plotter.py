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

def make_comparison(deets_list, labels=None, title="", which=0):
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


#make_comparison(['0-0-2-1-0', '2-0-2-1-0'], ['comm', 'no comm'], 'Effect of Adversary Recieving Communication', adversary=False)

#make_comparison(['0-0-2-1-0', '5-0-2-1-0', '10-0-2-1-0', '2-0-2-1-0'], ['comm', 'noisy comm', 'very noisy comm', 'no comm'], 'Effect of Noise on Adversary Performance', adversary=True)

#make_comparison(['2-1-2-1-0', '2-1-2-1-5'], ['3-channel comm', '5-channel comm'], 'Effect of Channel Width on Performance', adversary=False)

#make_comparison(['2-1-2-1-5', '2-1-4-0-5'], ['no sqrt', 'sqrt'], 'Effect of Sqrt with 5-channel comm', which=2)

#make_comparison(['2-1-2-1-0', '0-0-3-0-5'], ['no sqrt', 'sqrt'], 'Performance', which=2)

#make_comparison(['2-0-3-0-5-BIG', '2-0-3-0-5-BIG-NONOISE'], ['noise', 'nonoise'], 'Effect of Adversary Recieving Random Noise', which=0)

#make_comparison(['2-0-3-1-5-BIG-NOISE', '2-0-3-0-5-BIG'], ['corner goals', 'random goals'], 'Environment Modification on Listener Distance', which=2)

#make_comparison(['2-0-3-1-5-BIG-NOISE-NOCOMM', '2-0-3-1-5-BIG-NOISE-20', '2-0-3-1-5-BIG-NOISE-10', '2-0-3-1-5-BIG-NOISE-5', '2-0-3-1-5-BIG-NOISE-1', '2-0-3-1-5-BIG-NOISE-0'], 
#                ['no comm', 'very heavy noise', 'heavy noise', 'medium noise', 'light noise', 'clear comm'], 
#                'Communication Noise on Adversary Performance', which=1)

#make_comparison(['2-0-3-1-5-BIG-NOISE-NOCOMM', '2-0-3-1-5-BIG-NOISE-20', '2-0-3-1-5-BIG-NOISE-10', '2-0-3-1-5-BIG-NOISE-5', '2-0-3-1-5-BIG-NOISE-1', '2-0-3-1-5-BIG-NOISE-0'], 
#                ['no comm', 'very heavy noise', 'heavy noise', 'medium noise', 'light noise', 'clear comm'], 
#                'Adversary Comm Noise on Speaker/Listener Performance', which=0)

#make_comparison(['2-0-3-1-5-BIG-NOISE-NOCOMM', '2-0-3-1-5-BIG-NOISE-20', '2-0-3-1-5-BIG-NOISE-10', '2-0-3-1-5-BIG-NOISE-5', '2-0-3-1-5-BIG-NOISE-1', '2-0-3-1-5-BIG-NOISE-0'], 
#                ['no comm', 'very heavy noise', 'heavy noise', 'medium noise', 'light noise', 'clear comm'], 
#                'Adversary Communication Noise on Listener Distance', which=2)


#make_comparison(['2-0-3-1-5-BIG-NOISE-0', '0-0-3-1-5-BIG-COMM'], ['hierarchical', 'standard'], 
#                'Hierarchical Training on Listener Distance', which=2)

#make_comparison(['2-0-3-1-5-BIG-NOISE-0', '0-0-3-1-5-BIG-COMM'], ['hierarchical', 'standard'], 
#                'Hierarchical Training on Speaker/Listener Performance', which=0)

#make_comparison(['2-0-3-2-5-BIG-COMM'], ['moddedenv'], 
#                'Effect with Adversary Reward as Speaker Input', which=0)





def special_comparison(deets_list, labels=None, title=""):
    rewards_sl_list = [get_sl_rewards(deets) for deets in deets_list]
    rewards_adv_list = [get_adversary_rewards(deets) for deets in deets_list]
    rewards_list_list = [get_listener_distance(deets) for deets in deets_list]

    if labels is None:
        labels = deets_list

    #colora = ['tab:pink', 'tab:purple']
    #colorb = ['tab:cyan', 'tab:blue']
    colora = ['#f7c6c6', mcolors.cnames['darkred']]
    colorb = [mcolors.cnames['lightsteelblue'], mcolors.cnames['midnightblue']] 

    x = np.arange(60,120)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(len(rewards_sl_list)):
        ax1.plot(x, rewards_sl_list[i], label=labels[i], color=colora[(i+1)%2])

    ax1.legend()
    ax1.set_xlabel('# 1000s of episodes')
    ax1.set_ylabel('speaker/listener reward', color=colora[1])

    if title=="":
        ax1.set_title('Hierarchical Training on Speaker/Listener Performance')
    else:
        ax1.set_title(title)

    ax1.tick_params(axis='y', labelcolor=colora[1])
    ax1.set_ylim(-10, 50)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    for i in range(len(rewards_list_list)):
        ax2.plot(x, rewards_list_list[i], label=labels[i], color=colorb[(i+1)%2])
        
    ax2.set_ylabel('listener distance', color=colorb[1])
    ax2.tick_params(axis='y', labelcolor=colorb[1])

    ax2.set_ylim(-10, 50)

    leg = ax1.get_legend()
    leg.legendHandles[1].set_color('lightgray')
    leg.legendHandles[0].set_color('black')
    #plt.ylim([-10,30])
    plt.show()


special_comparison(['-2-0-3-1-5-BIG-NOISE-0', '-0-0-3-1-5-BIG-COMM'], ['hierarchical', 'standard'])

#special_comparison(['2-0-3-1-5-BIG-NOISE', '2-0-3-0-5-BIG'], ['corner goals', 'random goals'], 'Environment Modification on Listener Distance')