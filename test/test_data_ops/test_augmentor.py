import torch
import unittest

from data.data_augmentor.data_augmentor import lidar_augmentation

class TestDataAugmentator(unittest.TestCase):
    
    def test_data_augmentor_random_flip(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'random_flip_along_x')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)

        data_augmented = lidar_augmentation(lidar_data, 'random_flip_along_y')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
    def test_data_augmentor_global_ops(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'global_rotation')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'global_scaling')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
       
        
    def test_data_augmentor_random_translation(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'random_translation_along_x')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)

        data_augmented = lidar_augmentation(lidar_data, 'random_translation_along_y')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'random_translation_along_z')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)    
        
    
    def test_data_augmentor_random_local_translation(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'random_local_translation_along_x')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)

        data_augmented = lidar_augmentation(lidar_data, 'random_local_translation_along_y')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'random_local_translation_along_z')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)    
        
   
    def test_data_augmentor_global_frustum_dropout(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'global_frustum_dropout_top')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)

        data_augmented = lidar_augmentation(lidar_data, 'global_frustum_dropout_bottom')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'global_frustum_dropout_left')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'global_frustum_dropout_right')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)        
        
        
    def test_data_augmentor_local_ops(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'local_scaling')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'local_rotation')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        
    def test_data_augmentor_local_frustum_dropout(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'local_frustum_dropout_top')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)

        data_augmented = lidar_augmentation(lidar_data, 'local_frustum_dropout_bottom')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'local_frustum_dropout_left')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)
        
        data_augmented = lidar_augmentation(lidar_data, 'local_frustum_dropout_right')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)                  
  
        
    def test_data_augmentor_pyramid(self):
        points = torch.randn(2000, 4)
        gt_boxes = torch.randn(2000, 7)
        lidar_data = {"points": points, "gt_boxes": gt_boxes}

        data_augmented = lidar_augmentation(lidar_data, 'pyramid')
        assert not data_augmented['points'].equal(points)
        assert not data_augmented['gt_boxes'].equal(gt_boxes)   
        
        
if __name__ == '__main__':
    unittest.main()
