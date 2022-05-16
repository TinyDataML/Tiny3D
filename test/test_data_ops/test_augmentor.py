import torch
import unittest

from tiny3d.data.data_denoisor import lidar_augmentation


class TestDataAugmentator(unittest.TestCase):

    @classmethod
    def setUpData(cls):
        cls.points = torch.randn(2000, 4)
        cls.gt_boxes = torch.randn(2000, 7)
        cls.lidar_data = {"points": cls.points, "gt_boxes": cls.gt_boxes}

    # noinspection DuplicatedCode
    def test_data_augmentor(self):
        data_augmented = lidar_augmentation(cls.lidar_data, 'random_flip_along_x')
        assert data_augmented['points'].shape == torch.size([2000, 4])
        assert data_augmented['gt_boxes'].shape == torch.size([2000, 7])

        data_augmented = lidar_augmentation(cls.lidar_data, 'random_flip_along_y')
        assert data_augmented['points'].shape == torch.size([2000, 4])
        assert data_augmented['gt_boxes'].shape == torch.size([2000, 7])

if __name__ == '__main__':
    unittest.main()
