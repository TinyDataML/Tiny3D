import torch
import unittest

from data.data_simulator.data_simulator import lidar_simulator


class TestDataSimulator(unittest.TestCase):


    def test_data_simulator_foggy(self):
        points = torch.randn(2000, 4)
        lidar_data = {'points':points}
        data_simulated = lidar_simulator(lidar_data, 'foggy')
        print('=========== test_data_simulator_foggy ===========')
        print('input.type')
        print(type(points))
        print('output.type')
        print(type(data_simulated['points']))
        print('input.dtype')
        print(points.dtype)
        print('output.dtype')
        print(data_simulated['points'].dtype)
 
        assert not data_simulated['points'].equal(points)


if __name__ == '__main__':
    unittest.main()
