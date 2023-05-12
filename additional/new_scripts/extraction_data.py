
import torch
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import bottleneck as bn

def get_mean(data, cluster=-1):
    t = np.zeros(data.shape[0])
    div = 0
    for j in range(36):
        for k in range(72):
            if cluster == -1 or cluster_map[j,k] == cluster:
                t += data[:, j, k] * np.cos(np.radians(LAT[j]))
    
                div += np.cos(np.radians(LAT[j]))
    t /= div
    return t

def get_obs(cluster=-1):
    fn = data_dir + 'obs.nc'

    f = nc4.Dataset(fn, 'r')
    data = f.variables['temperature_anomaly'][:]

    return get_mean(data,cluster=cluster)

test = get_obs()

def get_pre_ind(type, model='IPSL', phys=1):
    
    dic = MODELS[model]
    giss_cond = ((type == 'hist-aer' and ((phys == 1 and i not in range(5, 10)) or (phys != 1 and i in range(5, 10)))) or \
                (type == 'historical' and ((phys == 1 and i < 10) or (phys != 1 and i >= 10)))) or \
                (type not in ['hist-aer', 'historical'] and i in range(5, 10))
    result = np.zeros((36,72))

    for i in range(dic[type]):

        if (model == 'GISS' and giss_cond) or model != 'GISS':
            
            fn = f'{data_dir}{model}_{type}_{str(i+1)}.nc'
            f = nc4.Dataset(fn, 'r')
        
            data = f.variables['tas'][0:50]
        
            result +=np.mean(data,axis=0)
        
    result /= dic[type]
    
    return result

def get_simu(type, simu, model='IPSL', cluster=-1, filtrage=False):
    
    if model == 'GISS':
        phys = 1
        i = simu
        if type == 'hist-aer':
            if i in range(6, 11):
                phys = 2
        elif type == 'historical':
            if i > 10:
                phys = 2
        pre_ind = get_pre_ind(type, model=model, phys=phys)

    else:
        pre_ind = get_pre_ind(type, model=model)

    fn = f'{data_dir}{model}_{type}_{str(i+1)}.nc'
    f = nc4.Dataset(fn, 'r')
    data = f.variables['tas'][50:]

    data = data - pre_ind
    result = get_mean(data, cluster=cluster)
    
    if(filtrage):
        if(type=='hist-GHG' or type=='hist-aer'):
    
            result = signal.filtfilt(b, a, result)
    return result

def get_data_forcage(type, model='IPSL', cluster=-1, filtrage=False):

    dic = MODELS[model]
    result = np.zeros((dic[type],115))
    for i in range(dic[type]):
        result[i] = get_simu(type, i+1, model, cluster, filtrage=filtrage)[0:115]
    
    return result

def get_metric(model='IPSL', cluster=-1, normalis=False, filtrage=False, metric='mean', as_tensor=False):

    aer = get_data_forcage('hist-aer', model=model, cluster=cluster, filtrage=filtrage)[:,0:115]
    ghg = get_data_forcage('hist-GHG', model=model, cluster=cluster, filtrage=filtrage)[:,0:115]
    nat = get_data_forcage('hist-nat', model=model, cluster=cluster, filtrage=filtrage)[:,0:115]
    historical = get_data_forcage('historical', model=model, cluster=cluster, filtrage=filtrage)[:,0:115]

    max_hist = np.max(np.mean(historical, axis=0))
    aer = aer / max_hist
    ghg = ghg / max_hist
    nat = nat / max_hist
    historical = historical / max_hist
    
    if normalis:
        if metric == 'std':
    
            aer = np.std(aer, axis=0)
            ghg = np.std(ghg, axis=0)
            nat = np.std(nat, axis=0)
            historical = np.std(historical, axis=0)

        elif metric == 'mean':

            aer = np.mean(aer, axis=0)
            ghg = np.mean(ghg, axis=0)
            nat = np.mean(nat, axis=0)
            historical = np.mean(historical, axis=0)

    if as_tensor:
        
        aer = torch.tensor(aer).float()
        ghg = torch.tensor(ghg).float(
        nat = torch.tensor(nat).float(
        historical = torch.tensor(historical).float(

    return aer, ghg, nat, historical

def get_data_set(model='IPSL', cluster=-1, normalis=False, filtrage=False):
    
    liste_max = []
    if model != 'ALL':

        aer, ghg, nat, historical = get_metric(model, cluster, normalis, filtrage, as_tensor=True)
        max_hist = np.max(np.mean(historical, axis=0))
        liste_max.append(max_hist)

    elif model == 'ALL':
        
        liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 
                        'HadGEM3', 'MIRO', 'ESM2', 'NorESM2','CESM2','GISS']

        aer = []
        ghg = []
        nat = []
        historical = []

        for model_curr in liste_models:

            aer_curr, ghg_curr, nat_curr, historical_curr = get_std(model_curr, cluster, normalis, filtrage, as_tensor=True)

            max_hist = torch.max(torch.mean(historical_curr, dim=0))
            liste_max.append(max_hist)

            aer.append(aer_curr)
            ghg.append(ghg_curr)
            nat.append(nat_curr)
            historical.append(historical_curr)

    return ghg, aer, nat, historical, np.array(liste_max)

def get_metric_data_set(model='IPSL', cluster=-1, normalis=False, filtrage=False, metric='mean'):
    
    if model != 'ALL':

        aer, ghg, nat, historical = get_metric(model, cluster, normalis, filtrage, metric)

        result = np.stack((ghg, aer, nat))        

    elif model == 'ALL':

        liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 
                        'HadGEM3', 'MIRO', 'ESM2', 'NorESM2','CESM2','GISS']
        result = []
        historical = []
        
        for model_curr in liste_models:

            aer_ipsl, ghg_ipsl, nat_ipsl, historical_ipsl = get_metric(model_curr, cluster, normalis, filtrage, metric)

            result_ipsl = np.stack((ghg_ipsl, aer_ipsl, nat_ipsl))
            result.append(result_ipsl)
            historical.append(historical_ipsl)

        result = np.mean(np.array(result), axis=0)
        historical = np.mean(np.array(historical), axis=0)
    
    return torch.tensor(result).unsqueeze(0), historical

