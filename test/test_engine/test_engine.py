import torch
import unittest

from deephub.detection_model import Pointpillars
from model.model_deployor.deployor_utils import create_input
from engine.engine_utils import Engine

pretrain_model = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
pcd = 'test/data_tobe_tested/kitti/kitti_000008.bin'
device = 'cuda:0'

data, model_inputs = create_input(pcd, 'kitti', 'pointpillars',
                                          device)
torch_model = Pointpillars()

class TestEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    # noinspection DuplicatedCode
    def test_engine_infer(self):
        # warp engine
        model = Engine(torch_model)
        # load pretrain model
        checkpoint = torch.load(pretrain_model)
        model.torch_model.load_state_dict(checkpoint["state_dict"])
        print('loading pretrain from:  ' + pretrain_model)

        # engine inference
        model.cuda()
        model.eval()

        predict = model(data['img_metas'][0], data['points'][0])

        # test
        assert len(predict['scores_3d']) != 0




if __name__ == '__main__':
    unittest.main()