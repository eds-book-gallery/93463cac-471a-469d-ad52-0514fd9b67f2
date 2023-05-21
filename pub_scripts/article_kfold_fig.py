import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from sklearn.metrics import mean_squared_error
import math

import extraction_data as extr

obs = np.array(extr.get_obs(cluster=-1))[0:115] * 1.06
max_obs = np.max(obs)
R = []

liste_models = [
	'CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS',
	'HadGEM3', 'MIRO', 'ESM2', 'NorESM2', 'CESM2', 'GISS', 'ALL'
]

model_true_name = [
	'CanESM5', 'CNRM-CM6-1', 'IPSL-CM6A-LR', 'ACCESS-ESM1-5',
	'BCC-CSM2-MR', 'FGOALS-g3', 'HadGEM3', 'MIROC6', 'MRI-ESM2.0',
	'NorESM2-LM', 'CESM2', 'GISS-E2-1-G', 'ALL'
]
# forc = 2
# for mod in range(12):
#
#
#     liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2', 'NorESM2',
#                     'CESM2', 'GISS', 'ALL']
#     model_true_name = ['CanESM5','CNRM-CM6-1','IPSL-CM6A-LR','ACCESS-ESM1-5',
#                        'BCC-CSM2-MR','FGOALS-g3','HadGEM3','MIROC6','ESM2','NorESM2-LM','CESM2','GISS-E2-1-G','ALL']
#     model= mod
#     ALL_data_orig = np.load('figures/Resul_model_inv/No_fil_pad_0/'+model_true_name[model]+'/model_10_parameters/cluster_-1/inver.npy') / max_obs
#
#     data_true_inv, inver_cible = extr.get_mean_data_set(liste_models[model], cluster=-1, normalis=False,
#                                                         filtrage=False)
#
#     std_true_inv, useless_std = extr.get_std_data_set(liste_models[model], cluster=-1, normalis=False, filtrage=False)
#     ghg_ueless, aer_useless, nat_useless, hist_cible, liste_useless = extr.get_data_set(liste_models[model],cluster=-1,normalis=False,filtrage=False)
#     ALL_data_orig *=liste_useless[0]
#
#
#     skill = []
#     for i in range(ALL_data_orig.shape[0]):
#         res = 0
#         ALL_data=ALL_data_orig[i]
#         moy_ghg = np.mean(ALL_data[:, 0], axis=0)
#         moy_aer = np.mean(ALL_data[:, 1], axis=0)
#         moy_nat = np.mean(ALL_data[:, 2], axis=0)
#         Pred = np.array([moy_ghg,moy_aer,moy_nat])
#
#
#
#         std_ghg = np.std(ALL_data[:, 0], axis=0)
#         std_aer = np.std(ALL_data[:, 1], axis=0)
#         std_nat = np.std(ALL_data[:, 2], axis=0)
#
#         print(data_true_inv.shape)
#         MSE = mean_squared_error(Pred[forc], data_true_inv[0,forc])
#
#         RMSE = math.sqrt(MSE)
#         skill.append(RMSE)
#
#     skill = np.array(skill)
#     print(np.mean(skill))
#     R.append(np.mean(skill))
#
# print(R)

fig, axs = plt.subplots(3, 4, figsize=([16, 12]), gridspec_kw={'width_ratios': [2, 2, 2, 2]})
for model in range(12):

	ALL_data_orig = np.load('figures/Resul_model_inv/No_fil_pad_0/' + model_true_name[
		model] + '/model_10_parameters/cluster_-1/inver.npy') / max_obs

	data_true_inv, inver_cible = extr.get_mean_data_set(
		liste_models[model],
		cluster=-1,
		normalis=False,
		filtrage=False
	)

	std_true_inv, useless_std = extr.get_std_data_set(
		liste_models[model],
		cluster=-1,
		normalis=False,
		filtrage=False
	)
	ghg_ueless, aer_useless, nat_useless, hist_cible, liste_useless = extr.get_data_set(
		liste_models[model],
		cluster=-1,
		normalis=False,
		filtrage=False
	)

	ALL_data_orig *= liste_useless[0]
	ALL_data = ALL_data_orig[1]
	moy_ghg = np.mean(ALL_data[:, 0], axis=0)
	moy_aer = np.mean(ALL_data[:, 1], axis=0)
	moy_nat = np.mean(ALL_data[:, 2], axis=0)

	std_ghg = np.std(ALL_data[:, 0], axis=0)
	std_aer = np.std(ALL_data[:, 1], axis=0)
	std_nat = np.std(ALL_data[:, 2], axis=0)

	i = model // 4
	j = model % 4

	axs[i, j].plot(np.array(hist_cible[0]), 'black', label="OBS")
	axs[i, j].plot(moy_ghg, 'red', label='Inversion of GHG')
	axs[i, j].fill_between(np.arange(115), moy_ghg - std_ghg, moy_ghg + std_ghg, facecolor='red', alpha=0.2)
	axs[i, j].plot(moy_aer, 'blue', label='Inversion of AER')
	axs[i, j].fill_between(np.arange(115), moy_aer - std_aer, moy_aer + std_aer, facecolor='blue', alpha=0.2)
	axs[i, j].plot(moy_nat, 'green', label='Inversion of NAT')
	axs[i, j].fill_between(np.arange(115), moy_nat - std_nat, moy_nat + std_nat, facecolor='green', alpha=0.2)

	axs[i, j].plot(data_true_inv[0, 0], 'purple', label='GHG')
	axs[i, j].fill_between(
		np.arange(115),
		data_true_inv[0, 0] - std_true_inv[0, 0],
		data_true_inv[0, 0] + std_true_inv[0, 0],
		facecolor='purple',
		alpha=0.2
	)
	axs[i, j].plot(data_true_inv[0, 1], 'darkblue', label='AER')
	axs[i, j].fill_between(
		np.arange(115),
		data_true_inv[0, 1] - std_true_inv[0, 1],
		data_true_inv[0, 1] + std_true_inv[0, 1],
		facecolor='darkblue',
		alpha=0.2
	)
	axs[i, j].plot(data_true_inv[0, 2], 'olive', label='NAT')
	axs[i, j].fill_between(
		np.arange(115),
		data_true_inv[0, 2] - std_true_inv[0, 2],
		data_true_inv[0, 2] + std_true_inv[0, 2],
		facecolor='olive',
		alpha=0.2
	)
	axs[i, j].set_title(model_true_name[model])
	axs[i, j].set_ylim((-1.4, 2.2))

	if (j != 0):
		axs[i, j].set_yticklabels([])
	else:
		axs[i, j].set_ylabel('°C')
	if (i != 2):
		axs[i, j].set_xticklabels([])
	else:
		axs[i, j].set_xlabel('Years')
		axs[i, j].set_xticks([0, 20, 40, 60, 80, 100, 114])
		axs[i, j].set_xticklabels(
			['1900', '1920', '1940', '1960', '1980', '2000', '2014'])
	axs[0, 0].legend()
	# axs[0,4].axis('off')
	# axs[1, 4].axis('off')
	# axs[2, 4].axis('off')

plt.show()
