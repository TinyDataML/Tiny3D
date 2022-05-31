import torch
import unittest


import data
import data.data_denoisor
from data.data_denoisor.data_denoisor import lidar_denoisor

class TestDataDenoisor(unittest.TestCase):

        
    # noinspection DuplicatedCode
    def test_data_denoisor_dmr(self):
        points = torch.randn(20000, 3)
        lidar_data = {'points':points}
        data_denoised = lidar_denoisor(lidar_data, 'dmr')
        assert not data_denoised['points'].equal(points)
        
    def test_data_denoisor_pcp(self):
        points = torch.randn(4, 500, 3)
        lidar_data = {'points':points}
        data_denoised = lidar_denoisor(lidar_data, 'pcp')
        assert not data_denoised['points'].equal(points)
if __name__ == '__main__':
    unittest.main()
