# This code is referenced from
# https://github.com/MartinHahner/LiDAR_fog_sim/
#
# Copyright Martin Hahner.
# All rights reserved.
#
# License: CC BY-NC 4.0

import os
import copy
import math
import pickle
import argparse

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.constants import speed_of_light as c     # in m/s

RNG = np.random.default_rng(seed=42)

# AVAILABLE_TAU_Hs = [20]
# LIDAR_FOLDERS = ['lidar_hdl64_strongest', 'lidar_hdl64_last']
# INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / 'integral_lookup_tables' / 'original


RNG = np.random.default_rng(seed=42)

AVAILABLE_TAU_Hs = [20]
LIDAR_FOLDERS = ['lidar_hdl64_strongest', 'lidar_hdl64_last']
INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / 'integral_lookup_tables' / 'original'


def get_available_alphas() -> List[float]:

    alphas = []

    for file in os.listdir(INTEGRAL_PATH):

        if file.endswith(".pickle"):

            alpha = file.split('_')[-1].replace('.pickle', '')

            alphas.append(float(alpha))

    return sorted(alphas)




class ParameterSet:

    def __init__(self, **kwargs) -> None:

        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.1                                    # in m              (displacement of transmitter and receiver)
        self.ROH_T = 0.01                               # in m              (radius of the transmitter aperture)
        self.ROH_R = 0.01                               # in m              (radius of the receiver aperture)
        self.GAMMA_T_DEG = 2                            # in deg            (opening angle of the transmitter's FOV)
        self.GAMMA_R_DEG = 3.5                          # in deg            (opening angle of the receiver's FOV)
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)

        # assert self.GAMMA_T_DEG != self.GAMMA_R_DEG, 'would lead to a division by zero in the calculation of R_2'
        #
        # self.R_1 = (self.D - self.ROH_T - self.ROH_R) / (
        #         np.tan(self.GAMMA_T / 2) + np.tan(self.GAMMA_R / 2))  # in m (see Figure 2 and Equation (11) in [1])
        # self.R_2 = (self.D - self.ROH_R + self.ROH_T) / (
        #         np.tan(self.GAMMA_R / 2) - np.tan(self.GAMMA_T / 2))  # in m (see Figure 2 and Equation (12) in [1])

        # R_2 < 10m in most sensors systems
        # co-axial setup (where R_2 = 0) is most affected by water droplet returns

        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_integral_dict(p: ParameterSet) -> Dict:

    alphas = get_available_alphas()

    alpha = min(alphas, key=lambda x: abs(x - p.alpha))
    tau_h = min(AVAILABLE_TAU_Hs, key=lambda x: abs(x - int(p.tau_h * 1e9)))

    filename = INTEGRAL_PATH / f'integral_0m_to_200m_stepsize_0.1m_tau_h_{tau_h}ns_alpha_{alpha}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)

    return integral_dict


def P_R_fog_hard(p: ParameterSet, pc: np.ndarray) -> np.ndarray:

    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)

    pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])

    return pc


def P_R_fog_soft(p: ParameterSet, pc: np.ndarray, original_intesity: np.ndarray, noise: int, gain: bool = False,
                 noise_variant: str = 'v1') -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = np.zeros(pc.shape)
    fog_mask = np.zeros(len(pc), dtype=bool)

    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)

    min_fog_response = np.inf
    max_fog_response = 0
    num_fog_responses = 0

    integral_dict = get_integral_dict(p)

    r_noise = RNG.integers(low=1, high=20, size=1)[0]
    r_noise = 10

    for i, r_0 in enumerate(r_zeros):

        # load integral values from precomputed dict
        key = float(str(round(r_0, 1)))
        # limit key to a maximum of 200 m
        fog_distance, fog_response = integral_dict[min(key, 200)]

        fog_response = fog_response * original_intesity[i] * (r_0 ** 2) * p.beta / p.beta_0

        # limit to 255
        fog_response = min(fog_response, 255)

        if fog_response > pc[i, 3]:

            fog_mask[i] = 1

            num_fog_responses += 1

            scaling_factor = fog_distance / r_0

            augmented_pc[i, 0] = pc[i, 0] * scaling_factor
            augmented_pc[i, 1] = pc[i, 1] * scaling_factor
            augmented_pc[i, 2] = pc[i, 2] * scaling_factor
            augmented_pc[i, 3] = fog_response

            # keep 5th feature if it exists
            if pc.shape[1] > 4:
                augmented_pc[i, 4] = pc[i, 4]

            if noise > 0:

                if noise_variant == 'v1':

                    # add uniform noise based on initial distance
                    distance_noise = RNG.uniform(low=r_0 - noise, high=r_0 + noise, size=1)[0]
                    noise_factor = r_0 / distance_noise

                elif noise_variant == 'v2':

                    # add noise in the power domain
                    power = RNG.uniform(low=-1, high=1, size=1)[0]
                    noise_factor = max(1.0, noise/5) ** power       # noise=10 => noise_factor ranges from 1/2 to 2

                elif noise_variant == 'v3':

                    # add noise in the power domain
                    power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                    noise_factor = max(1.0, noise*4/10) ** power    # noise=10 => ranges from 1/2 to 4

                elif noise_variant == 'v4':

                    additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]
                    new_dist = fog_distance + additive
                    noise_factor = new_dist / fog_distance

                else:

                    raise NotImplementedError(f"noise variant '{noise_variant}' is not implemented (yet)")

                augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor
                augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor
                augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor

            if fog_response > max_fog_response:
                max_fog_response = fog_response

            if fog_response < min_fog_response:
                min_fog_response = fog_response

        else:

            augmented_pc[i] = pc[i]

    if gain:
        max_intensity = np.ceil(max(augmented_pc[:, 3]))
        gain_factor = 255 / max_intensity
        augmented_pc[:, 3] *= gain_factor

    simulated_fog_pc = None

    if num_fog_responses > 0:
        fog_points = augmented_pc[fog_mask]
        simulated_fog_pc = fog_points

    info_dict = {'min_fog_response': min_fog_response,
                 'max_fog_response': max_fog_response,
                 'num_fog_responses': num_fog_responses}

    return augmented_pc, simulated_fog_pc, info_dict


def simulate_fog(p: ParameterSet, pc: np.ndarray, noise: int, gain: bool = False, noise_variant: str = 'v1',
                 hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = copy.deepcopy(pc)
    original_intensity = copy.deepcopy(pc[:, 3])

    info_dict = None
    simulated_fog_pc = None

    if hard:
        augmented_pc = P_R_fog_hard(p, augmented_pc)
    if soft:
        augmented_pc, simulated_fog_pc, info_dict = P_R_fog_soft(p, augmented_pc, original_intensity, noise, gain,
                                                                 noise_variant)

    return augmented_pc, simulated_fog_pc, info_dict


if __name__ == '__main__':

    args = parse_arguments()

    print('')
    print(f'using {args.n_cpus} CPUs')

    available_alphas = get_available_alphas()

    for lidar_folder in LIDAR_FOLDERS:

        src_folder = os.path.join(args.root_folder, lidar_folder)

        all_files = []

        for root, dirs, files in os.walk(src_folder, followlinks=True):
            assert (root == src_folder)
            all_files = sorted(files)

        all_paths = [os.path.join(src_folder, file) for file in all_files]

        for available_alpha in available_alphas:

            dst_folder = f'{src_folder}_CVL_beta_{available_alpha:.3f}'

            Path(dst_folder).mkdir(parents=True, exist_ok=True)

            print('')
            print(f'alpha {available_alpha}')
            print('')
            print(f'searching for point clouds in    {src_folder}')
            print(f'saving augmented point clouds to {dst_folder}')

            parameter_set = ParameterSet(alpha=available_alpha, gamma=0.000001)

            def _map(i: int) -> None:

                points = np.fromfile(all_paths[i], dtype=np.float32)
                points = points.reshape((-1, args.n_features))

                points, _, _ = simulate_fog(parameter_set, points, 10)

                lidar_save_path = os.path.join(dst_folder, all_files[i])
                points.astype(np.float32).tofile(lidar_save_path)

            n = len(all_files)

            with mp.Pool(args.n_cpus) as pool:

                l = list(tqdm(pool.imap(_map, range(n)), total=n))
