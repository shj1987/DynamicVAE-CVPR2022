""" 
Fun: compare the PID with other annealing method
KL vanishing comparison
Weight increas
Reconstruction error
"""

import os
import csv,json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import math
import importlib
from typing import Any


def _read_file(fileName,max_num,period=100):
    steps = []
    weight_avg = []
    rec_avg = []
    KL_avg = []
    elbo_avg = []

    elbo_period = []
    rec_period = []
    weight_period = []
    KL_period = []

    with open(fileName,"r") as f:
        for num,line in enumerate(f):
            if num == 0:
                continue
            arr = line.split()
            global_step = arr[0].replace('[',"").replace(']',"")
            step = int(global_step)
            if num == 1:
                step = 1
            ## KL loss
            # print("arr: ",arr)
            rec_loss = float(arr[1].split(':')[1])
            kl_loss = float(arr[2].split(':')[1])
            elbo = float(arr[3].split(':')[1])
            weight = float(arr[-1].split(':')[1])
            # print("weight: ",weight)
            weight_period.append(weight)
            rec_period.append(rec_loss)
            KL_period.append(kl_loss)
            elbo_period.append(elbo)

            # ## average result
            if (num) % period == 0 or num+1 >= max_num:
                steps.append(step)
                rec_avg.append(np.mean(rec_period))
                weight_avg.append(np.mean(weight_period))
                KL_avg.append(np.mean(KL_period))
                elbo_avg.append(np.mean(elbo_period))

                weight_period = []
                rec_period = []
                KL_period = []
                elbo_period = []

            if num+1 >= max_num:
                break

    return steps, weight_avg, rec_avg, KL_avg, elbo_avg
    

'''
Fun: plot figure
'''
def plot_figure(x, y, label_lst, x_title, location, fig_name, y_name):
    # fig = plt.figure()
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # axes= plt.axes()
    linewidth = 1.8 #linewidth
    colors = ['blue', 'black','red','orange','darkgreen','fuchsia','blue','grey','pink','grey','coral']
    markers = ['', '','','', '', '', '', '',' ^','v','d','+']
    # linestyles = ['-','--', '-','-.', '--', '--','--','--']*2
    linestyles = ['-','--', ':','-', '--', '--','--','--']*2
    n = len(y)
    print("# of y:",n)
    plt.plot(x, y, marker = markers[0], color = colors[0], linestyle=linestyles[0],\
            lw = linewidth, markersize=5, label = None)

    font2 = {'family' : 'Times New Roman','weight': 'normal','size': 14}
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
    plt.tick_params(labelsize = 17)
    plt.xlabel(x_title, fontsize = 17)  #we can use font 2
    plt.ylabel(y_name, fontsize = 17)
    plt.grid()
    plt.tight_layout()
    fig.savefig(fig_name,bbox_inches='tight',dpi = 500)
    plt.show()
    

def _create_folder(folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)


## main function
def main():
    ## compare the hit ratio
    folderName = 'figures'
    _create_folder(folderName)

    period = 20
    max_num = 10000
    beta_list = [130]
    
    ## for file name
    x_steps = []
    for beta in beta_list:
        # KL_each = []
        path = 'betaVAE_dsprites_beta' + str(beta)
        fileName = os.path.join(path, 'train.log')
        # fileName = 'train.log'
        steps, weight_avg, rec_avg, KL_avg, elbo_avg = _read_file(fileName, max_num, period)
        x_steps = steps
        # KL_each.append(KL_avg)
        ## KL loos
        kl_mean = KL_avg
        # print(kl_mean)
        label_lst = ['mnist']
        ## plot figure with shaded area
        location = 'best'
        x_title = 'training steps'
        ## rec loss
        figName = 'KL_'+ str(beta) + '.png'
        fig_name = os.path.join(folderName,figName)
        y_name = 'KL Divergence'
        plot_figure(x_steps, kl_mean, label_lst, x_title, location, fig_name, y_name)
        

    

if __name__ == '__main__':
    main()
    
    


