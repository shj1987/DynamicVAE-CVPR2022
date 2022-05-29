import os
import csv,json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import math
import importlib
from typing import Any
from tensorboard.backend.event_processing import event_accumulator


def _create_folder(folderName):
	if not os.path.exists(folderName):
		os.makedirs(folderName)


'''
Fun: plot figure
'''
def plot_figure(x, y, label_lst, x_title, fig_name, y_name, exp=None):
	# fig = plt.figure()
	fig, ax = plt.subplots()
	# axes= plt.axes()
	linewidth = 2.5 #linewidth
	# colors = ['blue', 'orange', 'darkgreen', 'yellow', 'lime', 'fuchsia', 'red', 'grey', 'pink', 'coral', 'black']
	colors = ['blue', 'orange', 'darkgreen', 'black', 'magenta', 'lime', 'grey', 'blue', 'pink', 'coral', 'pink']
	markers = ['', '','','', '', '', '', '']*4
	linestyles = ['-','-', '-','-', '-', '-','-','-']*5
	# edgecolors = ['#1B2ACC','#CC4F1B','#3F7F4C']
	# facecolors = ['#089FFF', '#FF9848', '#7EFF99']
	n = len(y)
	# print("# of y:",n)
	for i in range(n):
		# print(y[i][0])
		if x is None:
			plt.plot(y[i][:,1], y[i][:,2], marker = markers[i], color = colors[i], linestyle=linestyles[i], lw = linewidth, markersize=5, label = label_lst[i])
		else:
			plt.plot(x, y[i], marker = markers[i], color = colors[i], linestyle=linestyles[i], lw = linewidth, markersize=5, label = label_lst[i])
	
	font2 = {'family' : 'Times New Roman','weight': 'normal','size': 14}
	plt.tick_params(labelsize = 15)
	plt.xlabel(x_title, fontsize = 15)  #we can use font 2
	plt.ylabel(y_name, fontsize = 15)
	
	# plt.xticks(x, x)#show the X values
	# plt.xticks(np.arange(0, x[-1], 10000))
	# if x is not None:
	# 	ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'K'))
	### loc = "best",'upper left' = 2,'lower left'=3
	plt.legend(loc = 'best', prop={'size': 10.5})
	if exp is None:
		stepsize = 3
		# start, end = ax.get_xlim()
		ax.yaxis.set_ticks(np.arange(0, 20, stepsize))
	elif exp == 'RMIG':
		plt.ylim(0,1)
	elif exp == 'rec':
		stepsize = 10
		ax.yaxis.set_ticks(np.arange(0, 200, stepsize))
		plt.ylim(0,200)

	# plt.ylim(0, 19)
	# plt.title('Expected fusion error',fontsize = 14)
	plt.grid()
	plt.tight_layout()
	x_title = x_title.split()
	fig.set_size_inches(8, 6)
	fig.savefig(fig_name, bbox_inches='tight',dpi = 600)
	plt.show()


### plot kl of each dimension
def plot_kl_dimension(experiment):

	def _read_file(fileName,max_num,period=100):
		steps = []
		KL_avg = []
		KL_period = []
		total_KL_avg = []
		total_kl_period = []
		step = 0
		with open(fileName,"r") as f:
			for num,line in enumerate(f):
				arr = line.split()
				step += 20
				## KL loss
				total_kl = float(arr[0].split(':')[1])
				total_kl_period.append(total_kl)
				## wise element
				kl_loss = arr[1].split(':')[1]
				wise_KL = kl_loss.split(',')
				wise_KL = [float(k) for k in wise_KL]
				KL_period.append(wise_KL)
				# ## average result
				if (num) % period == 0 or num+1 >= max_num:
					steps.append(step)
					mean_total = np.mean(total_kl_period)
					mean_wise_kl = np.mean(KL_period,axis=0)
					# print(np.append(mean_wise_kl,mean_total))
					KL_avg.append(np.append(mean_wise_kl, mean_total))
					KL_period = []
					total_kl_period = []
				if num+1 >= max_num:
					break

		steps[0] = 1
		return steps, KL_avg
	
	## read record
	period = 100
	max_num = 1500000 / 20
	in_dir = './checkpoints/' + experiment
	fileName = os.path.join(in_dir, 'train.kl')
	steps, KL_avg = _read_file(fileName, max_num, period)
	
	## out file name
	out_dir = './results/' + experiment
	_create_folder(out_dir)

	## plot figure with shaded area
	x_title = 'Training steps'
	x_steps = steps
	# look at the gif to find the factors
	label_lst = ['z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','total KL']
	KL_trans = np.transpose(KL_avg)
	fig_name = os.path.join(out_dir, 'Sprites_KL_loss.eps')
	y_name = 'KL Divergence'
	plot_figure(x_steps, KL_trans, label_lst, x_title, fig_name, y_name)
	

### plot kl decomposition
def plot_kl_decomposition(experiment):
	## read log
	in_dir = './logs/' + experiment
	ea = event_accumulator.EventAccumulator(in_dir)
	ea.Reload()
	kld = np.array(ea.scalars.Items('crit/kld'))
	TC = np.array(ea.scalars.Items('crit/TC'))
	MI = np.array(ea.scalars.Items('crit/MI'))
	dKL = np.array(ea.scalars.Items('crit/dKL'))
	## out file
	out_dir = './results/' + experiment
	_create_folder(out_dir)
	## plot
	x_title = 'Training steps'
	fig_name = os.path.join(out_dir, 'Sprites_KL_decompose.eps')
	y_name = 'KL Divergence'
	plot_figure(None, [kld, TC, MI, dKL], ['kld', 'TC', 'MI', 'dKL'], x_title, fig_name, y_name)
	

def plot_rec_decomposition(experiment):
	## read log
	in_dir = './logs/' + experiment
	ea = event_accumulator.EventAccumulator(in_dir)
	ea.Reload()
	rec = np.array(ea.scalars.Items('crit/rec'))
	MI = np.array(ea.scalars.Items('crit/MI'))
	## out file
	out_dir = './results/' + experiment
	_create_folder(out_dir)
	## plot
	x_title = 'Training steps'
	fig_name = os.path.join(out_dir, 'Sprites_rec.eps')
	y_name = 'Reconstruction Loss'
	# plot_figure(None, [rec, MI], ['rec', 'MI'], x_title, fig_name, y_name)
	plot_figure(None, [rec], ['rec'], x_title, fig_name, y_name,'rec')

### plot RMIG
def plot_RMIG(experiment):
	# read record
	# out_dir = './results/' + experiment
	# _create_folder(out_dir)
	RMIG_file = os.path.join('./checkpoints/', experiment, 'RMIG.txt')
	steps = []
	shape = []
	scale = []
	rotation = []
	pos_x = []
	pos_y = []
	RIG4 = []
	RIG5 = []
	with open(RMIG_file, 'r') as f:
		for line in f:
			step, sh, sc, r, x, y, R4, R5 = line.split(' ')
			steps.append(int(step))
			shape.append(float(sh))
			scale.append(float(sc))
			rotation.append(float(r))
			pos_x.append(float(x))
			pos_y.append(float(y))
			RIG4.append(float(R4))
			RIG5.append(float(R5))
			
	steps = np.array(steps)
	shape = np.array(shape)
	scale = np.array(scale)
	rotation = np.array(rotation)
	pos_x = np.array(pos_x)
	pos_y = np.array(pos_y)
	RIG4 = np.array(RIG4)
	RIG5 = np.array(RIG5)
	# plot
	x_title = 'Training steps'
	out_dir = './results/' + experiment
	fig_name = os.path.join(out_dir, 'Sprites_RMIG.eps')
	y_name = 'RMIG'
	plot_figure(steps, [shape, scale, rotation, pos_x, pos_y, RIG4, RIG5], 
	['shape', 'scale', 'rotation', 'pos_x', 'pos_y', 'RIG4', 'RIG5'], x_title, fig_name, y_name, 'RMIG')


if __name__ == '__main__':
	experiment='dSprites-C20'
	print('>>>>>>>>>> begin [{}]'.format(experiment))
	# plot_kl_dimension(experiment)
	# print('>>>>>>>>>> done [KL dimension]')
	plot_kl_decomposition(experiment)
	print('>>>>>>>>>> done [KL decompose]')
	# plot_rec_decomposition(experiment)
	# print('>>>>>>>>>> done [Rec decompose]')
	# plot_RMIG(experiment)
	# print('>>>>>>>>>> done [RMIG]')

