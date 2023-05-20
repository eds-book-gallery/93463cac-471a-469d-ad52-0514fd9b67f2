from __future__ import annotations

import math
import pickle
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import extraction_data as extr  # fichier où sont rangés les codes permettant d'extraire les données
import inverse_model as inver  # fichier où sont rangés les codes permettant de faire l'inversion variationelle


# Fonction créant une arborescence pour stocker nos résultats
def mkdir_p(path: str) -> None:
	"""Creates a directory. equivalent to using mkdir -p on the command line"""
	if not os.path.exists(path):
		os.makedirs(path)

# créer la classe de data set de notre réseau de neurones
# all représente le modèle climatique étudié, si all vaut -1 on étudie tout les modèles en même temps
class MonDataset(Dataset):
	def __init__(self, ghg, aer, nat, historical, all=0, train_test: str = 'train'):

		self.ghg = ghg

		self.aer = aer
		self.nat = nat
		self.historical = historical
		self.train_test = train_test

		self.all = all

	def __len__(self):
		# On fixe arbitrairement une itération à l'étude de 50000 cas
		return 50000

	def __getitem__(self, item):

		if self.train_test == 'train':
			# on choisit au hasard un modèle qui n'est pas all
			while True:
				model = random.randint(0, len(self.ghg) - 1)
				if model != self.all: break

			# On choisit au hasard une simulation de chaque type du modèle
			ghg_max = self.ghg[model].shape[0] - 1
			aer_max = self.aer[model].shape[0] - 1
			nat_max = self.nat[model].shape[0] - 1
			hist_max = self.historical[model].shape[0] - 1

			indice_aer = random.randint(0, aer_max)
			indice_ghg = random.randint(0, ghg_max)
			indice_nat = random.randint(0, nat_max)
			indice_hist = random.randint(0, hist_max)
			# On crée les entrées et sortie
			X = torch.stack(
				(self.ghg[model][indice_ghg], self.aer[model][indice_aer], self.nat[model][indice_nat])).float()

			Y = self.historical[model][indice_hist].float()
		elif self.train_test == 'test':
			# même fonctionnement mais en ne prenant que le modèle all
			model = self.all
			if (model == -1):
				model = random.randint(0, len(self.ghg) - 1)
			ghg_max = self.ghg[model].shape[0] - 1
			aer_max = self.aer[model].shape[0] - 1
			nat_max = self.nat[model].shape[0] - 1
			hist_max = self.historical[model].shape[0] - 1

			indice_aer = random.randint(0, aer_max)
			indice_ghg = random.randint(0, ghg_max)
			indice_nat = random.randint(0, nat_max)
			indice_hist = random.randint(0, hist_max)

			X = torch.stack(
				(self.ghg[model][indice_ghg], self.aer[model][indice_aer], self.nat[model][indice_nat])).float()
			Y = self.historical[model][indice_hist].float()

		return X, Y, model


# on créer un data set simplifié pour l'inversion variationelle
class MonDataset_inverse(Dataset):
	def __init__(self, ghg, aer, nat):
		# ghg, aer, nat, historical = extr.get_data_set(model=model_data)
		self.ghg = ghg
		self.aer = aer
		self.nat = nat

	def __len__(self):
		return 100

	def __getitem__(self, item):
		ghg_max = self.ghg.shape[0] - 1
		aer_max = self.aer.shape[0] - 1
		nat_max = self.nat.shape[0] - 1

		indice_aer = random.randint(0, aer_max)
		indice_ghg = random.randint(0, ghg_max)
		indice_nat = random.randint(0, nat_max)
		X = torch.stack((self.ghg[indice_ghg], self.aer[indice_aer], self.nat[indice_nat])).float()

		return X


# fonction créant et entraiant les réseaux de neurones
# on donne en entrée les données et les parmètres d'apprentissage et renvoie les modèles appris
def train_model(data, data_test, lr=0.001, nb_epoch=100, taille=3, regularisation=-1):
	model = Net(taille, True)
	model_linear = Linear_mod()

	pytorch_total_params = math.fsum(p.numel() for p in model.parameters() if p.requires_grad)
	print(pytorch_total_params)

	criterion = nn.MSELoss()
	if (regularisation != -1):
		optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularisation)

		criterion2 = nn.MSELoss()
		optim2 = torch.optim.Adam(model_linear.parameters(), lr=lr, weight_decay=regularisation)
	else:

		optim = torch.optim.Adam(model.parameters(), lr=lr)

		criterion2 = nn.MSELoss()
		optim2 = torch.optim.Adam(model_linear.parameters(), lr=lr)

	Loss_tab = []
	Loss_test_tab = []
	Loss_tab_lin = []
	Loss_test_tab_lin = []

	for n_iter in range(nb_epoch):
		loss_total = 0
		loss_total_test = 0
		length = 0
		length_test = 0

		loss_total_2 = 0
		loss_total_test_2 = 0
		length_2 = 0
		length_test_2 = 0

		with torch.no_grad():
			for (x_test, y_test, mod) in data_test:
				y_hat_test = model(x_test)
				loss_test = criterion(y_hat_test.float(), y_test.float())
				loss_total_test += loss_test

				length_test += 1

				y_hat_test_2 = model_linear(x_test)
				loss_test_2 = criterion2(y_hat_test_2.float(), y_test.float())
				loss_total_test_2 += loss_test_2

				length_test_2 += 1

		for (x, y, mod) in data:
			y_hat = model(x)

			loss = criterion(y_hat.float(), y.float())
			loss.backward()
			optim.step()
			loss_total += loss
			optim.zero_grad()

			length += 1

			y_hat_2 = model_linear(x)
			loss_2 = criterion2(y_hat_2.float(), y.float())
			loss_2.backward()
			optim2.step()
			loss_total_2 += loss_2
			optim2.zero_grad()
			length_2 += 1

		print(f"Itérations {n_iter}: CNN:   loss train {loss_total / length} loss test {loss_total_test / length_test}")
		print(
			f"Itérations {n_iter}: Linear  loss train {loss_total_2 / length_2} loss test {loss_total_test_2 / length_test_2}")
		Loss_tab.append(loss_total.item() / length)
		Loss_test_tab.append(loss_total_test.item() / length_test)

		Loss_tab_lin.append(loss_total_2.item() / length)
		Loss_test_tab_lin.append(loss_total_test_2.item() / length_test)

	Loss_tab = np.array(Loss_tab)
	Loss_test_tab = np.array(Loss_test_tab)

	Loss_tab_lin = np.array(Loss_tab_lin)
	Loss_test_tab_lin = np.array(Loss_test_tab_lin)

	return model, Loss_tab, Loss_test_tab, model_linear, Loss_tab_lin, Loss_test_tab_lin


def train_and_plot(nom_dossier_root, clus=-1, all=0, normalis=False, denormalis=False, taille=3, filtrage=False,
                   regularisation=-1):
	liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2', 'NorESM2',
	                'CESM2', 'GISS', 'ALL']
	colors_model = ['red', 'blue', 'green', 'purple', 'gray', 'yellow', 'brown', 'orange', 'lawngreen', 'cyan', 'pink',
	                'olive']

	grad = 'new'
	print(clus)

	# on extrait les obs et
	obs = torch.tensor(extr.get_obs(cluster=clus))[0:115] * 1.06
	max_obs = torch.max(obs)
	obs = obs / max_obs

	nom_dossier = nom_dossier_root + 'cluster_' + str(clus) + '/'

	ghg, aer, nat, historical, liste_max = extr.get_data_set(cluster=clus, model='ALL', normalis=normalis,
	                                                         filtrage=filtrage)

	data = DataLoader(MonDataset(ghg, aer, nat, historical, type='train', all=all), shuffle=True, batch_size=BATCH_SIZE)
	data_test = DataLoader(MonDataset(ghg, aer, nat, historical, type='test', all=all), shuffle=True,
	                       batch_size=BATCH_SIZE)

	model, Loss_tab, Loss_test_tab, model_linear, Loss_tab_lin, Loss_test_tab_lin = train_model(data, data_test,
	                                                                                            taille=taille,
	                                                                                            regularisation=regularisation)

	if (nom_dossier != ''):
		mkdir_p('./figures/' + nom_dossier)

	torch.save(model, './figures/' + nom_dossier + 'model.pt')

	# inversion

	liste_res_for = []
	liste_res_cible = []

	if (all == -1):
		inver_cible = obs
		for i in range(len(liste_models) - 1):

			ghg, aer, nat, hist, liste = extr.get_data_set(liste_models[i], cluster=clus, normalis=normalis,
			                                               filtrage=filtrage)
			# ghg, aer, nat, hist, liste = extr.get_data_set('GISS', cluster=clus, normalis=normalis,
			# filtrage=filtrage)

			data_entree = DataLoader(MonDataset_inverse(ghg, aer, nat),
			                         batch_size=1)
			for pt_depart in data_entree:

				X, current = inver.model_inverse(pt_depart, torch.tensor(inver_cible), model)
				if denormalis:
					# liste_res_for.append(X.clone().detach().numpy()*liste_max[i])
					# liste_res_cible.append(current.clone().detach().numpy()*liste_max[i])
					if (all != -1):
						liste_res_for.append(X.clone().detach().numpy() * max_obs.item())
						liste_res_cible.append(current.clone().detach().numpy() * max_obs.item())
					else:
						liste_res_for.append(X.clone().detach().numpy() * liste_max[all])
						liste_res_cible.append(current.clone().detach().numpy() * liste_max[all])

				else:
					liste_res_for.append(X.clone().detach().numpy())
					liste_res_cible.append(current.clone().detach().numpy())
		liste_res_for = np.array(liste_res_for)[:, 0]
		liste_res_cible = np.array(liste_res_cible)[:, 0]
		with open('./figures/' + nom_dossier + 'inver.npy', 'wb') as f1:
			np.save(f1, liste_res_for)





	elif (all != -1):
		Result = []
		ghg_ueless, aer_useless, nat_useless, hist_cible, liste_useless = extr.get_data_set(liste_models[all],
		                                                                                    cluster=clus,
		                                                                                    normalis=normalis,
		                                                                                    filtrage=filtrage)

		for numb_inv in range(min(10, hist_cible.shape[0] - 1)):
			liste_res_for = []
			liste_res_cible = []
			inver_cible = hist_cible[numb_inv]
			for i in range(len(liste_models) - 1):
				if (i != all):

					ghg, aer, nat, hist, liste = extr.get_data_set(liste_models[i], cluster=clus, normalis=normalis,
					                                               filtrage=filtrage)
					# ghg, aer, nat, hist, liste = extr.get_data_set('GISS', cluster=clus, normalis=normalis,
					# filtrage=filtrage)

					data_entree = DataLoader(MonDataset_inverse(ghg, aer, nat),
					                         batch_size=1)
					for pt_depart in data_entree:

						X, current = inver.model_inverse(pt_depart, torch.tensor(inver_cible), model)
						if denormalis:
							# liste_res_for.append(X.clone().detach().numpy()*liste_max[i])
							# liste_res_cible.append(current.clone().detach().numpy()*liste_max[i])
							if (all == -1):
								liste_res_for.append(X.clone().detach().numpy() * max_obs.item())
								liste_res_cible.append(current.clone().detach().numpy() * max_obs.item())
							else:
								liste_res_for.append(X.clone().detach().numpy() * liste_max[all])
								liste_res_cible.append(current.clone().detach().numpy() * liste_max[all])

						else:
							liste_res_for.append(X.clone().detach().numpy())
							liste_res_cible.append(current.clone().detach().numpy())
			liste_res_for = np.array(liste_res_for)[:, 0]
			liste_res_cible = np.array(liste_res_cible)[:, 0]
			# liste_res_for = np.delete(liste_res_for,all,axis=0)
			Result.append(liste_res_for)
		Result = np.array(Result)
		with open('./figures/' + nom_dossier + 'inver.npy', 'wb') as f1:
			np.save(f1, Result)

	return Loss_test_tab


# #
class Net(nn.Module):

	def __init__(self, size_channel, bias):
		super(Net, self).__init__()

		self.tanh = nn.Tanh()
		self.conv1 = nn.Conv1d(3, size_channel, kernel_size=11, bias=bias, padding=5)
		self.conv2 = nn.Conv1d(size_channel, size_channel, kernel_size=11, bias=bias, padding=5)
		self.conv3 = nn.Conv1d(size_channel, 1, kernel_size=11, bias=bias, padding=5)

	def forward(self, X):
		x = self.conv1(X)
		x = self.tanh(x)
		x = self.conv2(x)
		x = self.tanh(x)
		x = x.float()
		x = self.conv3(x)[:, 0, :]

		return x


# modèle linéaire simple de comparaison
class Linear_mod(nn.Module):

	def __init__(self):
		super(Linear_mod, self).__init__()
		self.linear = nn.Linear(3, 1, bias=False)

	def forward(self, X):
		x = self.linear(X.transpose(1, 2))

		return x[:, :, 0]


# taille du batch pour l'apprentissage des réseaux de neurones
BATCH_SIZE = 100

liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2', 'NorESM2', 'CESM2',
                'GISS']
model_true_name = ['CanESM5', 'CNRM-CM6-1', 'IPSL-CM6A-LR', 'ACCESS-ESM1-5',
                   'BCC-CSM2-MR', 'FGOALS-g3', 'HadGEM3', 'MIROC6', 'ESM2', 'NorESM2-LM', 'CESM2', 'GISS-E2-1-G', 'ALL']
Loss_test_complete = []
taille = [10]
regul = [-1]
modules_tot = [-1]
for reg in regul:
	for tai in taille:

		for modu in range(11, 12):
			nom_dossier_root = '/Result/'

			loss_cur = train_and_plot(nom_dossier_root, all=modu, clus=-1, normalis=True, denormalis=True, taille=tai,
			                          filtrage=False, regularisation=reg)
			Loss_test_complete.append(loss_cur)
