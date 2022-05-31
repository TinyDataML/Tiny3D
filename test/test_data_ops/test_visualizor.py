import os
import torch
import unittest

import data
import data.data_simulator

from data.data_visualizor.data_visualizor import lidar_visualizor


class TestDataVisualizor(unittest.TestCase):

    def test_data_visulizor_open3d(self):
        points = torch.randn(2000, 4)
        pred_bboxes = torch.randn(2000, 7)
        gt_bboxes = torch.randn(2000, 7)
        lidar_data = {'points':points, 'pred_bboxes':pred_bboxes, 'gt_bboxes':gt_bboxes}
        data_simulated = lidar_visualizor(lidar_data, out_dir='./',filename='test_visualizor', method='open3d')
        
        assert os.path.exists('./test_visualizor')

if __name__ == '__main__':
    unittest.main()
