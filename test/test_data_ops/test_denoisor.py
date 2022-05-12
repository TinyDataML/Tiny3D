import torch
import unitest

from tiny3d.data.data_denoisor import lidar_denoisor

class TestDataDenoisor(unittest.TestCase):

    @classmethod
    def setUpData(cls):
        cls.points = torch.randn(2000,4)
        cls.lidar_data = {"points": cls.points}
        
    # noinspection DuplicatedCode
    def test_data_denoisor(self):
        data_denoised = lidar_denoisor(cls.lidar_data, 'dmr')
        assert data_denoised.shape == torch.size([2000,4])

if __name__ == '__main__':
    unittest.main()
