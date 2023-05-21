from __future__ import annotations

import os

# import bottleneck as bn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import netCDF4 as nc4  # TODO: look into the new alternative to netCDF4 if possible
# TODO: look into h5netcdf to aovid the netCDF-C libraries for compatibility
import numpy as np
import seaborn as sns
import torch

from scipy import signal
from scipy.io import loadmat


LIST_MODELS = [
	'CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS',
	'HadGEM3', 'MIRO', 'ESM2', 'NorESM2', 'CESM2', 'GISS'
]

b, a = signal.butter(20, 1 / 5, btype='lowpass')

data_dir = 'data_pre_ind_2'

cluster_map = np.zeros((36, 72), dtype=int)
cluster_map[-9:, :] = 1

fn = os.path.join(data_dir, 'obs.nc')

f = nc4.Dataset(fn, 'r')

data = f.variables['temperature_anomaly'][:]
LAT = f.variables['latitude'][:]


def get_mean(data: np.ndarray, cluster: int = -1):
	"""
	Calculate the mean value of the given data array.

	Parameters
	----------
	data : numpy.ndarray
		Input data array of shape (N, 36, 72).
	cluster : int, optional
		Cluster ID for filtering specific cluster data. Default is -1.

	Returns
	-------
	numpy.ndarray
		Mean value calculated from the input data array.

	Notes
	-----
	The mean value is calculated by considering the elements of the data array
	that meet the cluster filtering criteria. If `cluster` is -1, all elements
	are considered.

	The calculation is performed by summing the product of data values and the
	cosine of latitude values, and dividing by the sum of cosine of latitude
	values.
	"""

	# TODO: check the output type
	t = np.zeros((data.shape[0]))
	div = 0
	for j in range(36):
		for k in range(72):     # TODO: vectorize and check the range with the data array
			if cluster == -1 or cluster_map[j, k] == cluster:
				t += data[:, j, k] * np.cos(np.radians(LAT[j]))
				div += np.cos(np.radians(LAT[j]))
	t /= div
	return t


def get_obs(cluster: int = -1):
	"""
	Calculate the mean temperature anomaly from observed data.

	Parameters
	----------
	cluster : int, optional
		Cluster ID for filtering specific cluster data. Default is -1.

	Returns
	-------
	numpy.ndarray
		Mean temperature anomaly calculated from the observed data.
	"""
	fn = os.path.join(data_dir, 'obs.nc')
	f = nc4.Dataset(fn, 'r')
	data = f.variables['temperature_anomaly'][:]
	return get_mean(data, cluster=cluster)


test = get_obs()


# fonction extrayant la valeur pré-industrielle moyenne d'un modèle climatique
# function to get the pre-industrial mean value of a climatic model
def get_pre_ind(data_type: str, model: str = 'IPSL', phys: int = 1):
	model_dic = {
		'IPSL': {'hist-GHG': 10, 'hist-aer': 10, 'hist-nat': 10, 'historical': 32},
		'ACCESS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 30},
		'CESM2': {'hist-GHG': 3, 'hist-aer': 2, 'hist-nat': 3, 'historical': 11},
		'BCC': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3},
		'CanESM5': {'hist-GHG': 50, 'hist-aer': 30, 'hist-nat': 50, 'historical': 65},
		'FGOALS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 6},
		'GISS': {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19},
		'HadGEM3': {'hist-GHG': 4, 'hist-aer': 4, 'hist-nat': 4, 'historical': 5},
		'MIRO': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 50},
		'ESM2': {'hist-GHG': 5, 'hist-aer': 5, 'hist-nat': 5, 'historical': 7},     # TODO: check the historical value
		'NorESM2': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3},
		'CNRM': {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}
	}

	result = np.zeros((36, 72))

	if model == 'GISS':
		dic = model_dic[model]
		if data_type == 'hist-aer':
			divisor = 7 if phys == 1 else 5
			if phys == 1:
				idx = [i for i in range(dic[data_type]) if i not in {5, 6, 7, 8, 9}]
			else:
				idx = [i for i in range(dic[data_type]) if i in {5, 6, 7, 8, 9}]
		elif data_type == 'historical':
			divisor = 10 if phys == 1 else 9
			if phys == 1:
				idx = [i for i in range(dic[data_type]) if i < 10]
			else:
				idx = [i for i in range(dic[data_type]) if i >= 10]
		else:
			divisor = 5
			idx = [i for i in range(dic[data_type]) if i in {5, 6, 7, 8, 9}]
	else:
		idx = model_dic[model][data_type]
		divisor = model_dic[model][data_type]

	for i in range(idx):
		fn = os.path.join(data_dir, f"{model}_{data_type}_{i + 1}.nc")
		f = nc4.Dataset(fn, 'r')
		data = f.variables['tas'][0:50]
		result += np.mean(data, axis=0)
	result /= divisor
	return result


# fonction renvoyant 1 simulation
def get_simu(data_type: str, simu, model: str = 'IPSL', cluster: int = -1, filtering: bool = False):
	if model == 'GISS':
		i = simu
		if (data_type == 'hist-aer' and i in {6, 7, 8, 9, 10}) or (data_type == 'historical' and i > 10):
			phys = 2
		else:
			phys = 1
		pre_ind = get_pre_ind(data_type, model=model, phys=phys)
	else:
		pre_ind = get_pre_ind(data_type, model=model)

	fn = os.path.join(data_dir, f"{model}_{data_type}_{simu}.nc")
	f = nc4.Dataset(fn, 'r')
	data = f.variables['tas'][50:]

	data = data - pre_ind
	result = get_mean(data, cluster=cluster)
	if filtering:
		if data_type == 'hist-GHG' or data_type == 'hist-aer':
			result = signal.filtfilt(b, a, result)
	return result


# fonction renvoyant les simulations d'un certain type d'un modèle climatique
# function to get the simulations from a specific model
def get_data_forcage(data_type: str, model: str = 'IPSL', cluster: int = -1, filtrage: bool = False):
	model_dic = {
		'IPSL': {'hist-GHG': 10, 'hist-aer': 10, 'hist-nat': 10, 'historical': 32},
		'CNRM': {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30},
		'CESM2': {'hist-GHG': 3, 'hist-aer': 2, 'hist-nat': 3, 'historical': 11},
		'ACCESS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 30},
		'BCC': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3},
		'CanESM5': {'hist-GHG': 50, 'hist-aer': 30, 'hist-nat': 50, 'historical': 65},
		'FGOALS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 6},
		'GISS': {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19},
		'HadGEM3': {'hist-GHG': 4, 'hist-aer': 4, 'hist-nat': 4, 'historical': 5},
		'MIRO': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 50},
		'ESM2': {'hist-GHG': 5, 'hist-aer': 5, 'hist-nat': 5, 'historical': 5},     # TODO: check the historical value
		'NorESM2': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3}
	}

	result = np.zeros((model_dic[model][data_type], 115))
	for i in range(model_dic[model][data_type]):
		result[i] = get_simu(data_type, i + 1, model, cluster, filtering=filtrage)[0:115]

	return result


# fonction renvoyant le data-set entier traité
# function to get the full dataset and processes it
def get_data_set(model: str = 'IPSL', cluster: int = -1, normalis: bool = False, filtrage: bool = False):
	list_max = []

	if model != 'ALL':
		aer = get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		ghg = get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		nat = get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		historical = get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		max_hist = np.max(np.mean(historical, axis=0))
		list_max.append(max_hist)
		if normalis:
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist

	else:
		aer = []
		ghg = []
		nat = []
		historical = []

		for model_curr in LIST_MODELS:
			print(model_curr)
			aer_curr = torch.tensor(
				get_data_forcage('hist-aer', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115])
			ghg_curr = torch.tensor(
				get_data_forcage('hist-GHG', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115])
			nat_curr = torch.tensor(
				get_data_forcage('hist-nat', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115])
			historical_curr = torch.tensor(
				get_data_forcage('historical', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115])
			max_hist = torch.max(torch.mean(historical_curr, dim=0))
			list_max.append(max_hist)

			if normalis:
				aer_curr = aer_curr / max_hist
				ghg_curr = ghg_curr / max_hist
				nat_curr = nat_curr / max_hist
				historical_curr = historical_curr / max_hist

			aer.append(aer_curr)
			ghg.append(ghg_curr)
			nat.append(nat_curr)
			historical.append(historical_curr)

		return ghg, aer, nat, historical, np.array(list_max)

	return torch.tensor(ghg).float(), torch.tensor(aer).float(), torch.tensor(nat).float(), torch.tensor(
		historical).float(), np.array(list_max)


# renvoie les simulations moyenne de modèle climtique
# to get the mmean simulations of a model
def get_mean_data_set(model: str = 'IPSL', normalis: bool = False, cluster: int = -1, filtrage: bool = False):

	if model != 'ALL':
		aer = np.mean(get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		ghg = np.mean(get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		nat = np.mean(get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		historical = np.mean(get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage), axis=0)

		if normalis:
			max_hist = np.max(historical)
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist

	else:
		result = []
		historical = []
		for model_curr in LIST_MODELS:

			aer_ipsl = np.mean(get_data_forcage('hist-aer', model=model_curr, cluster=cluster, filtrage=filtrage), axis=0)
			ghg_ipsl = np.mean(get_data_forcage('hist-GHG', model=model_curr, cluster=cluster, filtrage=filtrage),	axis=0)
			nat_ipsl = np.mean(get_data_forcage('hist-nat', model=model_curr, cluster=cluster, filtrage=filtrage), axis=0)
			historical_ipsl = np.mean(get_data_forcage('historical', model=model_curr, cluster=cluster, filtrage=filtrage), axis=0)

			if normalis:
				max_hist = np.max(historical_ipsl)
				aer_ipsl = aer_ipsl / max_hist
				ghg_ipsl = ghg_ipsl / max_hist
				nat_ipsl = nat_ipsl / max_hist
				historical_ipsl = historical_ipsl / max_hist

			result_ipsl = np.stack((ghg_ipsl, aer_ipsl, nat_ipsl))
			result.append(result_ipsl)
			historical.append(historical_ipsl)

		result = np.array(result)
		result = np.mean(result, axis=0)

		historical = np.array(historical)
		historical = np.mean(historical, axis=0)

		return torch.tensor(result).unsqueeze(0), historical

	result = np.stack((ghg, aer, nat))
	return torch.tensor(result).unsqueeze(0), historical


# fonction renvoyant l"'écart moyen d'un modèle climatique
# get the std of a model
def get_std_data_set(model: str = 'IPSL', cluster: int = -1, normalis: bool = False, filtrage: bool = False):
	if model != 'ALL':

		aer = get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		ghg = get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		nat = get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		historical = get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		if normalis:
			max_hist = np.max(np.mean(historical, axis=0))
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist

		aer = np.std(aer, axis=0)
		ghg = np.std(ghg, axis=0)
		nat = np.std(nat, axis=0)
		historical = np.std(historical, axis=0)

	else:
		result = []
		historical = []
		for model_curr in LIST_MODELS:

			aer_ipsl = get_data_forcage('hist-aer', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			ghg_ipsl = get_data_forcage('hist-GHG', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			nat_ipsl = get_data_forcage('hist-nat', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			historical_ipsl = get_data_forcage(
				'historical',
				model=model_curr,
				cluster=cluster,
				filtrage=filtrage
			)[:, 0:115]
			if normalis:
				max_hist = np.max(np.mean(historical_ipsl, axis=0))
				aer_ipsl = aer_ipsl / max_hist
				ghg_ipsl = ghg_ipsl / max_hist
				nat_ipsl = nat_ipsl / max_hist
				historical_ipsl = historical_ipsl / max_hist

			aer_ipsl = np.std(aer_ipsl, axis=0)
			ghg_ipsl = np.std(ghg_ipsl, axis=0)
			nat_ipsl = np.std(nat_ipsl, axis=0)
			historical_ipsl = np.std(historical_ipsl, axis=0)

			result_ipsl = np.stack((ghg_ipsl, aer_ipsl, nat_ipsl))
			result.append(result_ipsl)
			historical.append(historical_ipsl)

		result = np.array(result)
		result = np.mean(result, axis=0)

		historical = np.array(historical)
		historical = np.mean(historical, axis=0)

		return torch.tensor(result).unsqueeze(0), historical

	result = np.stack((ghg, aer, nat))
	return torch.tensor(result).unsqueeze(0), historical


def get_map_compar(year1: int, year2: int, model: str = 'CNRM') -> None:
	results = []
	DATA_TYPES = ['hist-GHG', 'hist-aer', 'hist-nat', 'historical']
	dic = {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}
	for data_type in DATA_TYPES:
		result = np.zeros((36, 72))
		if data_type == 'hist-GHG':
			idx = [i for i in range(10) if i != 2]
		else:
			idx = [i for i in range(dic[data_type])]

		for i in idx:
			fn = os.path.join(data_dir, f"{model}_{data_type}_{i + 1}.nc")
			f = nc4.Dataset(fn, 'r')
			data = f.variables['tas'][year1 - 1850:year2 - 1850]
			result += np.mean(data, axis=0)

		result /= dic[data_type]
		results.append(result)

	results = np.array(results)
	res_sum = np.sum(results[0:3], axis=0)
	diff = results[3] - res_sum

	map_img = mpimg.imread('carte_terre.png')
	plt.figure(figsize=(14, 8))
	hmax = sns.heatmap(
		np.flipud(diff),
		alpha=0.5,
		annot=False,
		zorder=2,
		xticklabels=False,
		yticklabels=False,
		square=True,
		cmap='seismic'
	)
	hmax.imshow(
		map_img,
		aspect=hmax.get_aspect(),
		extent=hmax.get_xlim() + hmax.get_ylim(),
		zorder=1
	)

	plt.xticks([12, 24, 36, 48, 60], ['120°W', '60°W', '0°', '60°E', '120°E'])
	plt.yticks([8, 13, 18, 22, 28], ['60°N', '30°N', '0°', '30°S', '60°S'])
	plt.title(f'Difference between historical simulations and sum of forcing simulations.\nYears {year1}-{year2}')
	plt.tight_layout()
	plt.savefig(os.path.join('figures', f'diff_{year1}_{year2}_CNRM'))
	plt.show()


def plot_mean_simus() -> None:
	LIST_MODELS = ['CanESM5', 'CNRM', 'GISS', 'IPSL']
	nom_for = ['GHG', 'AER', 'Naturel']

	for i in range(3):
		for mod in LIST_MODELS:
			data, hist = get_mean_data_set(model=mod)

			# print(hist.shape)
			plt.plot(data[0, i], label=mod)
		plt.title(f'Simulation {nom_for[i]} mean for climatic models')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.xticks(
			[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
			['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']
		)
		plt.savefig('figures/simus_moyenne_' + str(nom_for[i]))
		plt.show()

	for mod in LIST_MODELS:
		data, hist = get_mean_data_set(model=mod)
		plt.plot(hist, label=mod)
	obs = get_obs()
	plt.plot(obs, label='Observations')
	plt.title('Simulation hist moyenne pour les modèles climatiques')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.xticks(
		[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
		['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']
	)
	plt.savefig('figures/simus_moyenne_hist')
	plt.show()

#
# plot_mean_simus()
