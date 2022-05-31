#import pylisa
import numpy as np
import torch

from .simulator_fog_utils import ParameterSet, simulate_fog

def lidar_simulator(lidar_data, method):
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
    points_original = lidar_data['points']
    points = points_original.numpy()

    if method == 'rainy':
        import pylisa
        lidar = pylisa.Lidar()  # lidar object
        water = pylisa.Water()  # material object
        rain = pylisa.MarshallPalmerRain()  # particle distribution model

        augm = pylisa.Lisa(lidar, water, rain)

        pcnew = augm.augment(points, 30)  # for a rain rate of 30 mm/hr
        points = augm.augment(pcnew)


    if method == "foggy":
        parameter_set = ParameterSet(alpha=0.5, gamma=0.000001)

        points, _, _ = simulate_fog(parameter_set, points, 10)

    lidar_data['points'] = torch.from_numpy(points).to(lidar_data['points'].device).type_as(points_original)

    return lidar_data
