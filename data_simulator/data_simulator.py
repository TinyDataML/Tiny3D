import pylisa

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

    if method == 'rainy':
        lidar = pylisa.Lidar()  # lidar object
        water = pylisa.Water()  # material object
        rain = pylisa.MarshallPalmerRain()  # particle distribution model

        augm = pylisa.Lisa(lidar, water, rain)

        pcnew = augm.augment(points, 30)  # for a rain rate of 30 mm/hr
        pcnew = augm.augment(pcnew)

        lidar_data['points'] = pcnew

    return lidar_data