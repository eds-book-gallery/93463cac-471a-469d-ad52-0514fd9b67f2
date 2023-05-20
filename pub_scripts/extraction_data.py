import numpy as np
import netCDF4 as nc4
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg
import seaborn as sns
import bottleneck as bn
from scipy import signal

b, a = signal.butter(20, 1 / 5, btype='lowpass')

data_dir = 'data_pre_ind_2/'

cluster_map = np.zeros((36, 72), dtype=int)
cluster_map[-9:, :] = 1

fn = data_dir + 'obs.nc'

f = nc4.Dataset(fn, 'r')

data = f.variables['temperature_anomaly'][:]
LAT = f.variables['latitude'][:]


# fonction faisant la moyenne spatiale d'une simulation
# function for the spatial norm of a simulation
def get_mean(data, cluster=-1):
	t = np.zeros((data.shape[0]))
	if (cluster == -1):
		div = 0
		for j in range(36):
			for k in range(72):
				t += data[:, j, k] * np.cos(np.radians(LAT[j]))

				div += np.cos(np.radians(LAT[j]))
		t /= div
		return t
	else:
		div = 0
		for j in range(36):
			for k in range(72):
				if (cluster_map[j, k] == cluster):
					t += data[:, j, k] * np.cos(np.radians(LAT[j]))
					div += np.cos(np.radians(LAT[j]))

		t /= div

		return t


# fonction extrayant les observations
# function to get the observations
def get_obs(cluster=-1):
	fn = data_dir + 'obs.nc'

	f = nc4.Dataset(fn, 'r')
	data = f.variables['temperature_anomaly'][:]

	return get_mean(data, cluster=cluster)


test = get_obs()


# fonction extrayant la valeur pré-industrielle moyenne d'un modèle climatique
# function to get the pre-industrial mean value of a climatic model
def get_pre_ind(type, model='IPSL', phys=1):
	if (model == 'IPSL'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 10, 'hist-aer': 10, 'hist-nat': 10, 'historical': 32}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'ACCESS'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 30}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'CESM2'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 2, 'hist-nat': 3, 'historical': 11}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'BCC'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'CanESM5'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 50, 'hist-aer': 30, 'hist-nat': 50, 'historical': 65}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'FGOALS'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 6}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'GISS'):
		if (type == 'hist-aer'):
			if (phys == 1):
				result = np.zeros((36, 72))
				dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}
				for i in range(dic[type]):
					if (i != 5 and i != 6 and i != 7 and i != 8 and i != 9):
						fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
						f = nc4.Dataset(fn, 'r')

						data = f.variables['tas'][0:50]

						result += np.mean(data, axis=0)
				result /= (7)
				return result

			else:
				result = np.zeros((36, 72))
				dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}
				for i in range(dic[type]):
					if (i == 5 or i == 6 or i == 7 or i == 8 or i == 9):
						fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
						f = nc4.Dataset(fn, 'r')

						data = f.variables['tas'][0:50]

						result += np.mean(data, axis=0)
				result /= 5
				return result
		elif (type == 'historical'):
			if (phys == 1):
				result = np.zeros((36, 72))
				dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}
				for i in range(dic[type]):
					if (i < 10):
						fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
						f = nc4.Dataset(fn, 'r')
						# print(i+1)
						# print(f.variables['tas'][:].shape)

						data = f.variables['tas'][0:50]

						result += np.mean(data, axis=0)
				result /= 10
				return result

			else:
				result = np.zeros((36, 72))
				dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}
				for i in range(dic[type]):
					if (i >= 10):
						fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
						f = nc4.Dataset(fn, 'r')
						# print(i+1)
						# print(f.variables['tas'][:].shape)

						data = f.variables['tas'][0:50]

						result += np.mean(data, axis=0)
				result /= 9
				return result
		else:
			result = np.zeros((36, 72))
			dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}
			for i in range(dic[type]):
				if (i == 5 or i == 6 or i == 7 or i == 8 or i == 9):
					fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
					f = nc4.Dataset(fn, 'r')

					data = f.variables['tas'][0:50]

					result += np.mean(data, axis=0)
			result /= 5
			return result


	elif (model == 'HadGEM3'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 4, 'hist-aer': 4, 'hist-nat': 4, 'historical': 5}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'MIRO'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 50}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'ESM2'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 5, 'hist-aer': 5, 'hist-nat': 5, 'historical': 7}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'NorESM2'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result

	elif (model == 'CNRM'):
		result = np.zeros((36, 72))
		dic = {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}
		for i in range(dic[type]):
			fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
			f = nc4.Dataset(fn, 'r')

			data = f.variables['tas'][0:50]

			result += np.mean(data, axis=0)
		result /= dic[type]
		return result


# fonction renvoyant 1 simulation
def get_simu(type, simu, model='IPSL', cluster=-1, filtrage=False):
	if model == 'GISS':
		phys = 1
		i = simu
		if (type == 'hist-aer'):
			if (i == 6 or i == 7 or i == 8 or i == 9 or i == 10):
				phys = 2
		elif (type == 'historical'):
			if (i > 10):
				phys = 2
		pre_ind = get_pre_ind(type, model=model, phys=phys)

	else:
		pre_ind = get_pre_ind(type, model=model)

	fn = data_dir + model + '_' + type + '_' + str(simu) + '.nc'
	f = nc4.Dataset(fn, 'r')
	# print(f.variables['tas'][:].shape)
	data = f.variables['tas'][50:]
	# print(data.shape)

	data = data - pre_ind
	result = get_mean(data, cluster=cluster)
	if (filtrage):
		if (type == 'hist-GHG' or type == 'hist-aer'):
			# result = bn.move_mean(result, window=5, min_count=1)
			result = signal.filtfilt(b, a, result)
	return result


#


# fonction renvoyant les simulations d'un certain type d'un modèle climatique
# function to get the simulations from a specific model
def get_data_forcage(data_type: str, model: str = 'IPSL', cluster: int = -1, filtrage: bool = False):
	dic = {}

	if model == 'IPSL':
		dic = {'hist-GHG': 10, 'hist-aer': 10, 'hist-nat': 10, 'historical': 32}

	elif model == 'CNRM':
		dic = {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}

	elif model == 'CESM2':
		dic = {'hist-GHG': 3, 'hist-aer': 2, 'hist-nat': 3, 'historical': 11}

	elif model == 'ACCESS':
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 30}

	elif model == 'BCC':
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3}

	elif model == 'CanESM5':
		dic = {'hist-GHG': 50, 'hist-aer': 30, 'hist-nat': 50, 'historical': 65}

	elif model == 'FGOALS':
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 6}

	elif model == 'GISS':
		dic = {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19}

	elif model == 'HadGEM3':
		dic = {'hist-GHG': 4, 'hist-aer': 4, 'hist-nat': 4, 'historical': 5}

	elif model == 'MIRO':
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 50}

	elif model == 'ESM2':
		dic = {'hist-GHG': 5, 'hist-aer': 5, 'hist-nat': 5, 'historical': 5}

	elif model == 'NorESM2':
		dic = {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3}

	result = np.zeros((dic[data_type], 115))
	for i in range(dic[data_type]):
		result[i] = get_simu(type, i + 1, model, cluster, filtrage=filtrage)[0:115]

	return result


# fonction renvoyant le data-set entier traité
# function to get the full dataset and processes it
def get_data_set(model='IPSL', cluster=-1, normalis=False, filtrage=False):
	liste_max = []
	if (model != 'ALL'):

		aer = get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		ghg = get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		nat = get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		historical = get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		max_hist = np.max(np.mean(historical, axis=0))
		liste_max.append(max_hist)
		if (normalis):
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist



	elif (model == 'ALL'):

		# liste_models = ['CanESM5', 'CNRM', 'GISS', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		#                'NorESM2','CESM2']
		liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		                'NorESM2', 'CESM2', 'GISS']

		aer = []
		ghg = []
		nat = []
		historical = []

		for model_curr in liste_models:
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
			liste_max.append(max_hist)

			if (normalis):
				aer_curr = aer_curr / max_hist
				ghg_curr = ghg_curr / max_hist
				nat_curr = nat_curr / max_hist
				historical_curr = historical_curr / max_hist

			aer.append(aer_curr)
			ghg.append(ghg_curr)
			nat.append(nat_curr)
			historical.append(historical_curr)

		return ghg, aer, nat, historical, np.array(liste_max)

	return torch.tensor(ghg).float(), torch.tensor(aer).float(), torch.tensor(nat).float(), torch.tensor(
		historical).float(), np.array(liste_max)


# renvoie les simulations moyenne de modèle climtique
# to get the mmean simulations of a model
def get_mean_data_set(model='IPSL', normalis=False, cluster=-1, filtrage=False):
	if (model != 'ALL'):

		aer = np.mean(get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		ghg = np.mean(get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		nat = np.mean(get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage), axis=0)
		historical = np.mean(get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage), axis=0)

		if (normalis):
			max_hist = np.max(historical)
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist



	elif (model == 'ALL'):

		# liste_models = ['CanESM5', 'CNRM', 'GISS', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		#                 'NorESM2','CESM2']
		liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		                'NorESM2', 'CESM2', 'GISS']
		result = []
		historical = []
		for model_curr in liste_models:

			aer_ipsl = np.mean(get_data_forcage('hist-aer', model=model_curr, cluster=cluster, filtrage=filtrage),
			                   axis=0)
			ghg_ipsl = np.mean(get_data_forcage('hist-GHG', model=model_curr, cluster=cluster, filtrage=filtrage),
			                   axis=0)
			nat_ipsl = np.mean(get_data_forcage('hist-nat', model=model_curr, cluster=cluster, filtrage=filtrage),
			                   axis=0)
			historical_ipsl = np.mean(
				get_data_forcage('historical', model=model_curr, cluster=cluster, filtrage=filtrage), axis=0)

			if (normalis):
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
def get_std_data_set(model='IPSL', cluster=-1, normalis=False, filtrage=False):
	if (model != 'ALL'):

		aer = get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		ghg = get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		nat = get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		historical = get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage)[:, 0:115]
		if (normalis):
			max_hist = np.max(np.mean(historical, axis=0))
			aer = aer / max_hist
			ghg = ghg / max_hist
			nat = nat / max_hist
			historical = historical / max_hist

		aer = np.std(aer, axis=0)
		ghg = np.std(ghg, axis=0)
		nat = np.std(nat, axis=0)
		historical = np.std(historical, axis=0)



	elif (model == 'ALL'):

		# liste_models = ['CanESM5', 'CNRM', 'GISS', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		#                 'NorESM2','CESM2']
		liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
		                'NorESM2', 'CESM2', 'GISS']
		result = []
		historical = []
		for model_curr in liste_models:

			aer_ipsl = get_data_forcage('hist-aer', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			ghg_ipsl = get_data_forcage('hist-GHG', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			nat_ipsl = get_data_forcage('hist-nat', model=model_curr, cluster=cluster, filtrage=filtrage)[:, 0:115]
			historical_ipsl = get_data_forcage('historical', model=model_curr, cluster=cluster, filtrage=filtrage)[:,
			                  0:115]
			if (normalis):
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


def get_map_compar(year1, year2, model='CNRM'):
	results = []
	types = ['hist-GHG', 'hist-aer', 'hist-nat', 'historical']
	for type in types:
		dic = {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}
		pre_ind = get_pre_ind(type)

		result = np.zeros((36, 72))
		# print(type)
		if (type == 'hist-GHG'):
			for i in range(2):
				fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
				f = nc4.Dataset(fn, 'r')
				data = f.variables['tas'][year1 - 1850:year2 - 1850]

				result += np.mean(data, axis=0)
			for i in range(3, 10, 1):
				fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
				f = nc4.Dataset(fn, 'r')
				data = f.variables['tas'][year1 - 1850:year2 - 1850]

				result += np.mean(data, axis=0)
		else:

			for i in range(dic[type]):
				fn = data_dir + model + '_' + type + '_' + str(i + 1) + '.nc'
				f = nc4.Dataset(fn, 'r')
				data = f.variables['tas'][year1 - 1850:year2 - 1850]

				result += np.mean(data, axis=0)
		result /= dic[type]
		# results.append(result - pre_ind)
		results.append(result)
	results = np.array(results)
	somme = np.sum(results[0:3], axis=0)
	diff = results[3] - somme

	map_img = mpimg.imread('carte_terre.png')
	plt.figure(figsize=(14, 8))
	hmax = sns.heatmap(np.flipud(diff), alpha=0.5, annot=False, zorder=2, xticklabels=False,
	                   yticklabels=False
	                   , square=True, cmap='seismic')
	hmax.imshow(map_img,
	            aspect=hmax.get_aspect(),
	            extent=hmax.get_xlim() + hmax.get_ylim(),
	            zorder=1)

	plt.xticks([12, 24, 36, 48, 60], ['120°W', '60°W', '0°', '60°E', '120°E'])
	plt.yticks([8, 13, 18, 22, 28], ['60°N', '30°N', '0°', '30°S', '60°S'])
	plt.title('Difference entre simulations historiques et sommes des simulations de forcages \n Années ' + str(
		year1) + '-' + str(year2))
	plt.tight_layout()
	plt.savefig('figures/diff_' + str(year1) + '_' + str(year2) + '_CNRM')
	plt.show()


def plot_mean_simus():
	liste_models = ['CanESM5', 'CNRM', 'GISS', 'IPSL']
	nom_for = ['GHG', 'AER', 'Naturel']

	for i in range(3):
		for mod in liste_models:
			data, hist = get_mean_data_set(model=mod)

			# print(hist.shape)
			plt.plot(data[0, i], label=mod)
		plt.title('Simulation ' + str(nom_for[i]) + ' moyenne pour les modèles climatiques')
		plt.legend(loc='best')
		plt.tight_layout()
		plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
		           ['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010',
		            '2020'])
		plt.savefig('figures/simus_moyenne_' + str(nom_for[i]))
		plt.show()

	for mod in liste_models:
		data, hist = get_mean_data_set(model=mod)
		plt.plot(hist, label=mod)
	obs = get_obs()
	plt.plot(obs, label='Observations')
	plt.title('Simulation hist moyenne pour les modèles climatiques')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
	           ['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'])
	plt.savefig('figures/simus_moyenne_hist')
	plt.show()

#
# plot_mean_simus()
