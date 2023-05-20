import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import extraction_data_2 as extr

ALL_data = np.load('figures/ALL/inver.npy')
print(ALL_data.shape)
data = np.sort(ALL_data[:, 2, 93])
data = data[30:1200 - 30]
print(data[0])
print(data[-1])
exit(1)
moy_ghg = np.mean(ALL_data[:, 0], axis=0)
moy_aer = np.mean(ALL_data[:, 1], axis=0)
moy_nat = np.mean(ALL_data[:, 2], axis=0)

std_ghg = np.std(ALL_data[:, 0], axis=0)
std_aer = np.std(ALL_data[:, 1], axis=0)
std_nat = np.std(ALL_data[:, 2], axis=0)
obs = np.array(extr.get_obs(cluster=-1))[0:115] * 1.06

C = 0
if (C == 1):
	plt.plot(np.array(obs), 'black', label="OBS")
	plt.plot(moy_ghg, 'red', label='GHG')
	plt.fill_between(np.arange(115), moy_ghg - std_ghg, moy_ghg + std_ghg, facecolor='red', alpha=0.2)
	plt.plot(moy_aer, 'blue', label='AER')
	plt.fill_between(np.arange(115), moy_aer - std_aer, moy_aer + std_aer, facecolor='blue', alpha=0.2)
	plt.plot(moy_nat, 'green', label='NAT')
	plt.fill_between(np.arange(115), moy_nat - std_nat, moy_nat + std_nat, facecolor='green', alpha=0.2)
	plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
	           ['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'])

	plt.ylabel('K')
	plt.xlabel('Years')
	plt.ylim((-0.75, 1.5))
	plt.legend()
	plt.show()

	fig, axs = plt.subplots(2, 1)

	axs[0].boxplot(ALL_data[:, :, 92])

	axs[1].boxplot(ALL_data[:, :, 114])
	axs[0].set_xticklabels([])
	axs[1].set_xticklabels(['GHG', 'AER', 'NAT'])

	axs[0].set_title('Year 1992')
	axs[1].set_title('Year 2014')

	axs[0].set_ylim(-1.2, 2)
	axs[1].set_ylim(-1.2, 2)
	plt.show()

	fig, axs = plt.subplots(3, 1)

	data = ALL_data[:, 0, 92]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)

	axs[0].plot(dist_space, kde(dist_space), c='red', label='1992')
	data = ALL_data[:, 0, 114]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)

	axs[0].plot(dist_space, kde(dist_space), c='blue', label='2014')
	axs[0].set_title('GHG')

	data = ALL_data[:, 1, 92]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)

	axs[1].plot(dist_space, kde(dist_space), c='red')
	data = ALL_data[:, 1, 114]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)
	axs[1].plot(dist_space, kde(dist_space), c='blue')
	axs[1].set_title('AER')

	data = ALL_data[:, 2, 92]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)
	axs[2].plot(dist_space, kde(dist_space), c='red')
	data = ALL_data[:, 2, 114]
	kde = gaussian_kde(data)
	dist_space = linspace(min(data), max(data), 100)
	axs[2].plot(dist_space, kde(dist_space), c='blue')
	axs[2].set_title('NAT')

	fig.legend()
	plt.tight_layout()
	plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.8), gridspec_kw={'width_ratios': [3, 1, 1]})


def box_plot(data, edge_color, fill_color):
	bp = axs[1].boxplot(data, patch_artist=True)

	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(bp[element], color=edge_color)

	for patch in bp['boxes']:
		patch.set(facecolor=fill_color)

	return bp


axs[0].plot(np.array(obs), 'black', label="OBS")
axs[0].plot(moy_ghg, 'red', label='GHG')
axs[0].fill_between(np.arange(115), moy_ghg - std_ghg, moy_ghg + std_ghg, facecolor='red', alpha=0.2)
axs[0].plot(moy_aer, 'blue', label='AER')
axs[0].fill_between(np.arange(115), moy_aer - std_aer, moy_aer + std_aer, facecolor='blue', alpha=0.2)
axs[0].plot(moy_nat, 'green', label='NAT')
axs[0].fill_between(np.arange(115), moy_nat - std_nat, moy_nat + std_nat, facecolor='green', alpha=0.2)
axs[0].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
axs[0].set_xticklabels(
	['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'])

axs[0].set_ylabel('K')
axs[0].set_xlabel('Years')


def histo(data, ind, color_model):
	bin = int(round((np.max(data) - np.min(data)) * 10)) + 1
	print(data.shape)
	axs[ind].hist(data, bins=bin, stacked=True, color=color_model, alpha=0.5, orientation="horizontal")


histo(ALL_data[:, 0, 93], 1, color_model='red')
histo(ALL_data[:, 1, 93], 1, color_model='blue')
histo(ALL_data[:, 2, 93], 1, color_model='green')
histo(ALL_data[:, 0, 114], 2, color_model='red')
histo(ALL_data[:, 1, 114], 2, color_model='blue')
histo(ALL_data[:, 2, 114], 2, color_model='green')

axs[1].set_title('1993')
axs[2].set_title('2014')

axs[0].set_ylim((-1.5, 2))
axs[1].set_ylim((-1.5, 2))
axs[2].set_ylim((-1.5, 2))

plt.show()
exit(1)

#
# bp1 = box_plot([ALL_data[:,0,93],ALL_data[:,0,114]], 'red', 'tan')
# bp2 = box_plot([ALL_data[:,1,93],ALL_data[:,1,114]], 'blue', 'cyan')
# bp3 = box_plot([ALL_data[:,2,92],ALL_data[:,2,114]], 'green','olive')
#
#
# axs[1].set_xticklabels(['1993','2014'])
# axs[1].set_ylim((-0.75, 1.5))
#
# data = ALL_data[:, 0, 93]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[2].plot(kde(dist_space),dist_space,  c='red')
# data = ALL_data[:, 0, 114]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[3].plot(kde(dist_space),dist_space,  c='red')
#
#
# data = ALL_data[:, 1, 93]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[2].plot(kde(dist_space),dist_space,  c='blue')
# data = ALL_data[:, 1, 114]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[3].plot(kde(dist_space),dist_space,  c='blue')
#
#
# data = ALL_data[:, 2, 92]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[2].plot(kde(dist_space),dist_space,  c='green')
# data = ALL_data[:, 2, 114]
# kde = gaussian_kde(data)
# dist_space = linspace(min(data), max(data), 100)
# axs[3].plot(kde(dist_space),dist_space,  c='green')
#
# axs[2].set_ylim((-0.75, 1.5))
# axs[3].set_ylim((-0.75, 1.5))
# axs[2].set_title('1993')
# axs[3].set_title('2014')
#
# axs[4].axis('off')
# fig.legend()
# plt.tight_layout()
#
#
#
# colors_model = ['red','blue','green','purple','gray','yellow','brown','orange','lawngreen','cyan','pink','olive']
# model_true_name = ['CanESM5','CNRM-CM6-1','IPSL-CM6A-LR','ACCESS-ESM1-5',
#                    'BCC-CSM2-MR','FGOALS-g3','HadGEM3','MIROC6','ESM2','NorESM2-LM','CESM2','GISS-E2-1-G']
#
# def histo(data,ind,ind2):
#     bin = int(round((np.max(data) - np.min(data)) * 10)) +1
#     data_2 = []
#     for i in range(12):
#         data_2.append(data[i*100:(i+1)*100])
#     data = np.transpose(np.array(data_2))
#     print(data.shape)
#     if(ind==0 and ind2==0):
#         axs[ind,ind2].hist(data,bins=bin,stacked=True,color=colors_model,label=model_true_name)
#     else:
#         axs[ind,ind2].hist(data, bins=bin, stacked=True, color=colors_model)
#
#
# bins = 10
# fig, axs = plt.subplots(2,4,figsize=(8.4,4.8))
# histo(ALL_data[:,0,93],0,0)
# histo(ALL_data[:,1,93],0,1)
# histo(ALL_data[:,2,93],0,2)
# histo(ALL_data[:,0,114],1,0)
# histo(ALL_data[:,1,114],1,1)
# histo(ALL_data[:,2,114],1,2)
# axs[0,0].set_title('GHG')
# axs[1,0].set_title('GHG')
# axs[0,1].set_title('1993 \nAER')
# axs[1,1].set_title('2014 \nAER')
# axs[0,2].set_title('NAT')
# axs[1,2].set_title('NAT')
# axs[0,3].axis('off')
# axs[1,3].axis('off')
# #fig.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
# fig.legend()
# plt.tight_layout()
#
# plt.show()
