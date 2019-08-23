import gym
gym.logger.set_level(40)
import numpy as np
from usable_functions import *
import time
import cartesian
from agent_gp import agent_gp_c
from sklearn import gaussian_process, preprocessing
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic

max_num_of_episodes = 10

#Initializing everything
h = np.array([0,0])

#initializing environment and agents
env = gym.make('LunarLanderContinuous-v2')

# Human Model
H_kernel =  ConstantKernel(.008)*RBF(length_scale=.2) + WhiteKernel(noise_level=1e-7) 
H_gp_1 = gaussian_process.GaussianProcessRegressor(kernel=H_kernel, optimizer = None, normalize_y=False)
H_gp_2 = gaussian_process.GaussianProcessRegressor(kernel=H_kernel, optimizer = None, normalize_y=False)
H_obs_ar_1 = np.zeros((1,9))
H_obs_ar_2 = np.zeros((1,9))
H_h_ar_1 = np.zeros((1))
H_h_ar_2 = np.zeros((1))

# Policy
P_kernel =  ConstantKernel(.008)*Matern(length_scale=.4, nu=1.5) + WhiteKernel(noise_level=1e-7) 
P_gp_1 = gaussian_process.GaussianProcessRegressor(kernel=P_kernel, optimizer = None, normalize_y=False)
P_gp_2 = gaussian_process.GaussianProcessRegressor(kernel=P_kernel, optimizer = None, normalize_y=False)
P_obs_ar_1 = np.zeros((1,8))
P_obs_ar_2 = np.zeros((1,8))
P_actions_1 = np.zeros((1))
P_actions_2 = np.zeros((1))

# Sparsification parameterization
thresshold_std = .5 * np.sqrt(P_kernel.get_params()['k1__k1__constant_value']) # * P_kernel.get_params()['k1__k2__length_scale'] # 1 * sqrt(Constant_kernel) * length_scale (standard deviation)


for i_episode in range(max_num_of_episodes):

	print('Starting episode', i_episode+1)

	# Initiate environement, agent and oracle
	observation = env.reset()  

	# Cumulative reward set to 0	
	reward_c = 0.
	feedback_e_1 = 0
	feedback_e_2 = 0

	for t in range(env._max_episode_steps+1):

		observation_normalized = observation.reshape(1,-1)
		action_1, action_1_std = np.clip(P_gp_1.predict(observation_normalized.reshape(1,-1), return_std=True),-1,1)
		action_2, action_2_std = np.clip(P_gp_2.predict(observation_normalized.reshape(1,-1), return_std=True),-1,1)
		action = np.concatenate((action_1, action_2))

		time.sleep(.01)

		# Activate render
		env.render()

		# Set step and recieve either human or oracle feedback
		obs_action_1 = np.concatenate((observation.reshape(1,-1),action_1.reshape(1,-1)),axis=1)
		obs_action_1_normalized = obs_action_1.reshape(1,-1)
		H_est_1,H_std_1 = H_gp_1.predict(obs_action_1_normalized.reshape(1,-1), return_std=True)


		obs_action_2 = np.concatenate((observation.reshape(1,-1),action_2.reshape(1,-1)),axis=1)
		obs_action_2_normalized = obs_action_2.reshape(1,-1)
		H_est_2,H_std_2 = H_gp_2.predict(obs_action_2_normalized.reshape(1,-1), return_std=True)


		if h[0] in [-1,1,-2,2]:
			if h[0] == -1:
				print('left')
			if h[0] == 1:
				print('right')
			if h[0] == -2:
				print('up')
			if h[0] == 1:
				print('down')

			#Learning rate
			if h[0] in [-1,1]:

				feedback_e_1 += 1


				H_h_ar_1 = np.concatenate((H_h_ar_1,[h[0]]))
				H_obs_ar_1 = np.concatenate((H_obs_ar_1,  obs_action_1),axis=0)
				H_gp_1.fit(H_obs_ar_1,H_h_ar_1)


				hm_lr_1 			= H_std_1 + action_1_std + .1
				new_action_1 		= action_1 + hm_lr_1 * h[0]
				

				if action_1_std < thresshold_std and P_obs_ar_1.shape[0]>10:
					index = np.argmax(P_kernel.__call__(P_obs_ar_1,observation_normalized.reshape(1,-1)))
					P_actions_1[index] = new_action_1
					P_obs_ar_1[index] = observation.reshape(1,-1)

				else:
					P_obs_ar_1 		= np.concatenate((P_obs_ar_1,observation.reshape(1,-1)),axis=0)
					P_actions_1 		= np.concatenate((P_actions_1,new_action_1.reshape(1)),axis=0)
				


			if h[0] in [-2,2]:

				feedback_e_2 += 1

				h[0] = np.sign(h[0])


				H_h_ar_2 = np.concatenate((H_h_ar_2,[h[0]]))
				H_obs_ar_2 = np.concatenate((H_obs_ar_2,  obs_action_2),axis=0)
				H_gp_2.fit(H_obs_ar_2,H_h_ar_2)


				hm_lr_2 = H_std_2 + action_2_std + .1 

				new_action_2 		= action_2 + hm_lr_2 * h[0] 
			
				if action_2_std < thresshold_std and P_obs_ar_2.shape[0]>10:
					index = np.argmax(P_kernel.__call__(P_obs_ar_2,observation_normalized.reshape(1,-1)))
					P_actions_2[index] = new_action_2
					P_obs_ar_2[index] = observation.reshape(1,-1)

				else:
					P_obs_ar_2 		= np.concatenate((P_obs_ar_2,observation.reshape(1,-1)),axis=0)
					P_actions_2 	= np.concatenate((P_actions_2,new_action_2.reshape(1)),axis=0)
				
			
			P_gp_1.fit(P_obs_ar_1,P_actions_1)
			P_gp_2.fit(P_obs_ar_2,P_actions_2)

		observation, h, done, _ = env.step(action) 

		if done:

			break

		reward_c = reward_c + h[1]


	print('\tCumulative FB: ', feedback_e_1 + feedback_e_2, '\tReward:', reward_c)

env.close()


