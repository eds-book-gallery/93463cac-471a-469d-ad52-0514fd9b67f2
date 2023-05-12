
import numpy as np
import netCDF4 as nc4
from scipy import signal

b, a = signal.butter(20, 1/5, btype='lowpass')

data_dir = '/home/users/acz25/CI2023-RC-team2/data_pre_ind_2/'

cluster_map = np.zeros((36,72), dtype=int)
cluster_map[-9:,:] = 1

fn = data_dir + 'obs.nc'

f = nc4.Dataset(fn, 'r')

data = f.variables['temperature_anomaly'][:]
LAT = f.variables['latitude'][:]

MODELS = {'IPSL': {'hist-GHG': 10, 'hist-aer': 10, 'hist-nat': 10, 'historical': 32},
          'ACCESS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 30},
          'CESM2': {'hist-GHG': 3, 'hist-aer': 2, 'hist-nat': 3, 'historical': 11},
          'BCC': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3},
          'CanESM5': {'hist-GHG': 50, 'hist-aer': 30, 'hist-nat': 50, 'historical': 65},
          'FGOALS': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 6},
          'GISS': {'hist-GHG': 10, 'hist-aer': 12, 'hist-nat': 20, 'historical': 19},
          'HadGEM3': {'hist-GHG': 4, 'hist-aer': 4, 'hist-nat': 4, 'historical': 5},
          'MIRO': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 50},
          'ESM2': {'hist-GHG': 5, 'hist-aer': 5, 'hist-nat': 5, 'historical': 7},
          'NorESM2': {'hist-GHG': 3, 'hist-aer': 3, 'hist-nat': 3, 'historical': 3},
          'CNRM': {'hist-GHG': 9, 'hist-aer': 10, 'hist-nat': 10, 'historical': 30}
         }