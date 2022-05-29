import torch
import unittest

from data.data_loader.data_loader import lidar_loader

class TestDataDenoisor(unittest.TestCase):

    
    def test_data_loader_bin(self):
        data_loaded=None
        lidar_data_path = 'test/date_tobe_tested/nuscenes/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
        data_loaded = lidar_loader(lidar_data_path, None, 'pcd.bin')
        
        print('output.shape')
        print(data_loaded.shape)
        assert data_loaded.any() != None
        
        
if __name__ == '__main__':
    unittest.main()
