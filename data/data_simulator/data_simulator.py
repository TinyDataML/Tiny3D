import pylisa
import numpy as np

from simulation_fog_utils import ParameterSet, simulate_fog

def lidar_simulation(lidar_data, method):
    """
        Use different extreme weather lidar simulation methods to augment lidar data.
        Args:
            lidar_data: dict
            method: str
        return:
            simulated lidar_data
        Reference:
            https://github.com/velatkilic/LISA
    """
    points = lidar_data['points']
    points = points.numpy()

    if method == 'rainy':
        lidar = pylisa.Lidar()  # lidar object
        water = pylisa.Water()  # material object
        rain = pylisa.MarshallPalmerRain()  # particle distribution model

        augm = pylisa.Lisa(lidar, water, rain)

        pcnew = augm.augment(points, 30)  # for a rain rate of 30 mm/hr
        points = augm.augment(pcnew)


    if method == "foggy":
        parameter_set = ParameterSet(alpha=0.5, gamma=0.000001)

        points, _, _ = simulate_fog(parameter_set, points, 10)

    lidar_data['points'] = points

    return lidar_data
