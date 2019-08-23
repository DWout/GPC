import numpy as np 
from scipy.signal import argrelextrema
from time import strftime, localtime
import time
import os, shutil


def make_archive():
	path = os.path.basename(os.getcwd()) #Current folder
	archive_from = os.getcwd()
	destination = '/home/dwout/Documents/thesis_all/data/' + path + '/archives/'
	os.makedirs(os.path.dirname(destination), exist_ok=True)
	name = strftime("%Y%m%d_%H%M%S",localtime())
	format = 'zip'
	shutil.make_archive(name, format, archive_from)
	time.sleep(.2)
	shutil.move('%s.%s'%(name,format), destination)
	print('Backup made in backup folder with name', name)

	return name

def acquisition_function(obs,actions,amount,delta_t,std = [],random = False):

	# Random from uncertainty sequence

	if std != []:

		index_lm = argrelextrema(std, np.greater)[0]
		np.random.shuffle(index_lm)
		index_lm = [index_lm]

		# Random 
		if random == True:
			index_lm = np.random.choice(range(1,len(actions[0])), len(index_lm[0]), replace=False)
			index_lm = [index_lm]

	else:

		index_lm = np.random.choice(range(1,len(actions[0])), amount, replace=False)

	# First 3 

	# index_lm = argrelextrema(std, np.greater)

	try:
		index_lm[0][0]
	except:
		amount = 0
		print('WARNING: No local maxima found, no inquiries will be executed')


	
	init_values = np.empty((0,3))
	action_seq = np.empty((0,delta_t*2))
	high_uncertain = index_lm[0][0:amount]
	start_indexx = np.clip(index_lm[0][0:amount]-delta_t,0,len(actions[0])-1-2*delta_t)

	start_indexx[start_indexx<0]=0

	for i in range(amount):
		try:
			init_values = np.concatenate((init_values,obs[start_indexx[i].astype(int)].reshape(-1,3)),axis=0)
			action_seq = np.concatenate((action_seq, actions[0][start_indexx[i].astype(int):start_indexx[i].astype(int)+delta_t*2].reshape(1,-1)), axis=0)
		except:
			print('WARNING: \'Not enought local maxima found, limited to: ', i, '\'')
			break
	

	return init_values, action_seq, high_uncertain, start_indexx

def acquisition_function_rand(std,obs,actions,amount,delta_t):

	init_values = np.empty((0,3))
	action_seq = np.empty((0,delta_t*2))
	high_uncertain = index_lm[0][0:amount]
	start_indexx = np.clip(index_lm[0][0:amount]-delta_t,0,len(actions[0])-1-2*delta_t)
# Print iterations progress

def printProgressBar (iteration, total, hu, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)█
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (hu / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	bar = list(bar)
	bar[int(length*hu//total)] = '#'
	bar = ''.join(bar)
	print('\r','%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r', flush = True)

def CountDown (text, seconds):
	print('\n', text, end = '\n')
	for i in range(seconds):
		if i != seconds-1:
			print('\r',20*str(seconds-i), end = '', flush=True)
			time.sleep(1)
		else:
			print('\r',20*str(seconds-i), end = '\n', flush=True)
			time.sleep(1)