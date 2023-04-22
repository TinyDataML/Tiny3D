import os
import sys
import torch
from deephub.detection_model import Pointpillars, Centerpoint

def main():
    model = Pointpillars()

    # print(model.name_parameters())

if __name__ == '__main__':
    main()