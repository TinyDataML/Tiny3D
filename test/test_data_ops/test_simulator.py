import torch
import unitest

from tiny3d.data.data_simulator import lidar_simulator


class TestDataSimulator(unittest.TestCase):

    @classmethod
    def setUpData(cls):
        cls.points = torch.randn(2000, 4)

    # noinspection DuplicatedCode
    def test_data_simulator_rainy(self):
        data_simulated = lidar_simulation(cls.lidar_data, 'rainy')
        assert data_simulated['points'].shape == torch.size([2000, 4])

    def test_data_simulator_foggy(self):
        data_simulated = lidar_simulation(cls.lidar_data, 'foggy')
        assert data_simulated['points'].shape == torch.size([2000, 4])


if __name__ == '__main__':
    unittest.main()
