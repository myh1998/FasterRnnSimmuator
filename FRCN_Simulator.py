import sys
import warnings
import random

warnings.filterwarnings("ignore", message="Failed to load image Python extension")
# warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from Simulator.RBFleX import RBFleX
from Simulator.Computation import DL2
from Simulator.scale_sim.scale import scale as ScaleSim

import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
# Multi Objective Bayesian Optimization
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize, standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, \
    qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
import time
from Simulator.defines.single_run import mainstage

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.double
else:
    DEVICE = "cpu"
    DTYPE = torch.double

DEVICE = "cpu"
DTYPE = torch.double

tkwargs = {
    "dtype": DTYPE,
    "device": DEVICE,
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class FRCN_Simulator:

    def __init__(self, IN_H, IN_W, optimized_components,
                 #  dataset="ImageNet", score_input_batch=8, img_root="/Users/tomomasayamasaki/Library/CloudStorage/OneDrive-SingaporeUniversityofTechnologyandDesign/SUTD/Life_of_University/Lab/#5Research-FRCNSim/Program/Dataset/ILSVRC2012_img_val_for_ImageFolder",
                 dataset="cifar100", score_input_batch=8, img_root="",
                 benchmark="sss", benchmark_root="/Volumes/UPAN/RBFleX-NAS/COFleX/NATS-sss-v1_0-50262-simple",
                 n_hyper=10, ref_score=-1000,
                 iters=20, mc_samples=128, acqu_algo="qNEHVI", batch_size=4, num_restarts=10, raw_samples=512,
                 n_init_size=10, mapping="ws",
                 Hardware_Arch="DL2"):

        print()
        print('+++++++++++ Configuration +++++++++++')
        with torch.no_grad():
            #######################################
            # General config    
            #######################################
            print("[General config]")
            self.n_init_size = n_init_size
            self.image = torch.ones(1, 3, IN_H, IN_W)
            print("\tInput resolution: [{}, {}]".format(IN_H, IN_W))
            print("\tAcquation function: {}".format(acqu_algo))  # 用于多目标贝叶斯优化（MOBO）的采集函数。它专门设计用于处理具有噪声的观测数据
            print("\tBatch size for initial data generation: {}".format(n_init_size))
            print("\tDevice: {}".format(DEVICE))

            #######################################
            # Config for NAS     
            #######################################
            print("[RBFleX config]")
            self.batch_size_score = n_hyper  # The number of batch images for hyperparameter detection algorithm
            self.benchmark = benchmark
            print("\tbatch images for HDA: {}".format(self.batch_size_score))
            # High-Dimensional Data Analysis"（高维数据分析）
            # 在 RBFleX 配置中提到的 "batch images for HDA" 指在进行高维数据分析时所使用的图像批处理。例如，将一批图像（每批包含10个图像）用于模型训练或测试

            #######################################
            # Config for DSE  Design Space Exploration”（设计空间探索）
            #######################################
            print("[DSE config]")
            self.optimized_comp = optimized_components
            self.opt_architecture = 0
            self.hardware_components_values = {"X1": 0, "X2": 0, "X3": 0, "X4": 0, "X5": 0, "X6": 0}
            self.opt_params = [k for k, v in self.optimized_comp.items() if v == 0] # ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'] at Begin
            self.not_opt_params = [k for k, v in self.optimized_comp.items() if v != 0] # [] at Begin
            for key in self.not_opt_params:
                self.hardware_components_values[key] = (self.optimized_comp[key], self.optimized_comp[key])
            for key in self.opt_params:
                self.hardware_components_values[key] = (20, 36) # {'X1': (20, 36), 'X2': (20, 36), 'X3': (20, 36), 'X4': (20, 36), 'X5': (20, 36), 'X6': (20, 36)}
            self.Hardware_Arch = Hardware_Arch
            if self.Hardware_Arch == "DL2":
                self.Num_HWopt = 6
            elif self.Hardware_Arch == "ScaleSim":
                self.Num_HWopt = 6
            elif self.Hardware_Arch == "DeFiNES":
                self.Num_HWopt = 6  # Number of HW varies to be optimized
            print("Hardware Architecture: {}".format(self.Hardware_Arch))
            print('\tTo be optimized: {}'.format(self.opt_params))
            print('\tFixed HW params: {}'.format(self.not_opt_params))
            if self.Hardware_Arch == "DL2":
                if not mapping in ["rs", "ws"]:
                    print("[ERROR] mapping for DL2 supports only [rs, ws].")
            elif self.Hardware_Arch == "ScaleSim":
                if not mapping in ["os", "ws", "is"]:
                    print("[ERROR] mapping for systolic array supports only [os, ws, is].")
            elif self.Hardware_Arch == "DeFiNES":
                if not mapping in ["os", "ws", "is"]:
                    # print("[ERROR] mapping for systolic array supports only [os, ws, is].")
                    pass
            print('\tMapping: {} stationary'.format(mapping))

            #######################################
            #  Config for Multiple object baysian optimazation
            #######################################
            # NAS
            if self.benchmark == "sss":
                self.nas_dim = 5
                self.nas_obj = 1
                self.sf_lower_bound = 8
                self.sf_upper_bound = 64
                self.sf_bounds = torch.stack(
                    [torch.ones(self.nas_dim, **tkwargs), 8.0 * torch.ones(self.nas_dim, **tkwargs)])
                self.sf_norm = torch.stack([self.sf_lower_bound * torch.ones(self.nas_dim, **tkwargs),
                                            self.sf_upper_bound * torch.ones(self.nas_dim, **tkwargs)])
                self.sf_standard_bounds = torch.stack(
                    [torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])
            elif self.benchmark == "201":
                from design_space.models import get_search_spaces
                from design_space.policy import PolicyTopology
                self.space_201 = get_search_spaces('cell', 'nas-bench-201')
                self.policy = PolicyTopology(self.space_201)
                self.nas_dim = 6
                self.nas_obj = 1
                self.sf_lower_bound = 0
                self.sf_upper_bound = 4
                self.sf_bounds = torch.stack(
                    [torch.zeros(self.nas_dim, **tkwargs), 4.0 * torch.ones(self.nas_dim, **tkwargs)])
                self.sf_norm = self.sf_bounds
                self.sf_standard_bounds = torch.stack(
                    [torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])

            self.hd_dim = len(self.opt_params)
            if self.Hardware_Arch == "DL2":
                self.hd_obj = 2  # [energy and cycle]
                self.SCORE_IDX = 0
                self.ENERGY_IDX = 1
                self.CYCLE_IDX = 2
            elif self.Hardware_Arch == "ScaleSim":
                self.hd_obj = 1  # [cycle]
                self.SCORE_IDX = 0
                self.CYCLE_IDX = 1
            if self.Hardware_Arch == "DeFiNES":
                self.hd_obj = 2  # [energy and cycle]
                self.SCORE_IDX = 0
                self.ENERGY_IDX = 1
                self.CYCLE_IDX = 1
            self.mobo_dim = self.nas_dim + self.hd_dim  # how many obj to be optimized
            self.mobo_obj = self.nas_obj + self.hd_obj  # how many output
            self.ref_point = torch.zeros(self.mobo_obj, **tkwargs)  # reference point
            self.ref_point[0] = ref_score
            if self.Hardware_Arch == "DL2":
                self.hd_lower_bound = [20, 20]  # [0]: for PE array [1]: for memory
                self.hd_upper_bound = [60, 60]  # [0]: for PE array [1]: for memory
            elif self.Hardware_Arch == "ScaleSim":
                self.hd_lower_bound = [1, 10]  # [0]: for PE array [1]: for memory
                self.hd_upper_bound = [64, 512]  # [0]: for PE array [1]: for memory
            elif self.Hardware_Arch == "DeFiNES":
                self.hd_lower_bound = [1, 10]  # [0]: for PE array [1]: for memory
                self.hd_upper_bound = [64, 512]  # [0]: for PE array [1]: for memory
            nk = 0
            self.hd_bounds = [[0] * self.Num_HWopt, [0] * self.Num_HWopt]
            for k, v in self.optimized_comp.items():
                if v == 0:
                    if k == "X1" or k == "X2": #config the PE size x*y
                        self.hd_bounds[0][nk] = self.hd_lower_bound[0] # 1
                        self.hd_bounds[1][nk] = self.hd_upper_bound[0] # 64
                    else:
                        self.hd_bounds[0][nk] = self.hd_lower_bound[1]
                        self.hd_bounds[1][nk] = self.hd_upper_bound[1]
                else:
                    self.hd_bounds[0][nk] = v
                    self.hd_bounds[1][nk] = v
                nk += 1
            self.hd_bounds = torch.tensor(self.hd_bounds, **tkwargs)

            nk = 0
            self.hd_standard_bounds = torch.zeros(2, self.hd_dim, **tkwargs)
            self.hd_standard_bounds[1] = 1
            for k, v in self.optimized_comp.items():
                if not v == 0:
                    self.hd_standard_bounds[0][nk] = 1
                nk += 1

            nk = 0
            self.hd_norm = [[0] * self.Num_HWopt, [0] * self.Num_HWopt]
            for k, v in self.optimized_comp.items():
                if v == 0:
                    self.hd_norm[0][nk] = self.hd_lower_bound[1]
                    self.hd_norm[1][nk] = self.hd_upper_bound[1]
                    if k == "X1" or k == "X2":
                        self.hd_norm[0][nk] = self.hd_lower_bound[0]
                        self.hd_norm[1][nk] = self.hd_upper_bound[0]
                else:
                    self.hd_norm[0][nk] = 0
                    self.hd_norm[1][nk] = v
                nk += 1
            self.hd_norm = torch.tensor(self.hd_norm, **tkwargs)

            self.bounds = torch.cat((self.sf_bounds, self.hd_bounds), 1)
            self.bounds_fornorm = torch.cat((self.sf_norm, self.hd_norm), 1)
            self.bounds_forstard = torch.cat((self.sf_standard_bounds, self.hd_standard_bounds), 1)

            self.BATCH_SIZE = batch_size
            self.NUM_RESTARTS = num_restarts if not SMOKE_TEST else 2
            self.RAW_SAMPLES = raw_samples if not SMOKE_TEST else 4
            self.N_BATCH = iters if not SMOKE_TEST else 10  # number of iteration
            self.MC_SAMPLES = mc_samples if not SMOKE_TEST else 16
            self.acqu_algo = acqu_algo

            #######################################
            # Initilize RBFleX, DSE, and Estimator    
            #######################################
            self.RBFleX = RBFleX(dataset, score_input_batch, img_root, benchmark, benchmark_root, self.batch_size_score)
            if self.Hardware_Arch == "DL2":
                self.AutoDSE = DL2(self.optimized_comp, mapping)
            elif self.Hardware_Arch == "ScaleSim":
                self.AutoDSE = ScaleSim(mapping)
            elif self.Hardware_Arch == "DeFiNES":
                self.AutoDSE = None
            self.mapping = mapping

    def _rbflex_and_dse(self, image, candi_network, candi_hardparams):
        #######################
        #      RBFleX-NAS     
        #######################
        network_score, backbone, layers_dist, uid = self.RBFleX.run(image=image, arch=candi_network)

        #######################
        #         DSE         
        #######################
        if self.Hardware_Arch == "DL2":
            energy, cycle, latency = self.AutoDSE.run(image, layers_dist, candi_hardparams)
        elif self.Hardware_Arch == "ScaleSim":
            candi_hardparams[5] = 10  # bandwidth from ScaleSim
            print("==> Current selected HW arch: PE size_x {:.0f}, PE size_y {:.0f}, Mem for Ifmaps {:.0f}, "
                  "Mem for Ofmaps {:.0f}, Mem for W {:.0f}, Input Bandwidth {:.0f}".format(candi_hardparams[0],
                                                                                     candi_hardparams[1],
                                                                                     candi_hardparams[2],
                                                                                     candi_hardparams[3],
                                                                                     candi_hardparams[4],
                                                                                     candi_hardparams[5]))
            # sys.exit()
            energy, cycle, latency = self.AutoDSE.run(layers_dist, candi_hardparams)
            print("==> Current Energy & Latency: {:.6f} mJ, {:.6f} million cycles".format(energy, cycle))
        elif self.Hardware_Arch == 'DeFiNES':
            energy, cycle = 0, 0
            # set_uid(uid)
            print("==> Current selected HW arch: PE size_x {:.0f}, PE size_y {:.0f}, Mem for Ifmaps {:.0f}, "
                  "Mem for Ofmaps {:.0f}, Mem for W {:.0f}, Input Bandwidth {:.0f}".format(candi_hardparams[0],
                                                                                           candi_hardparams[1],
                                                                                           candi_hardparams[2],
                                                                                           candi_hardparams[3],
                                                                                           candi_hardparams[4],
                                                                                           candi_hardparams[5]))
            candi_hardparams[5] = 8  # default bandwidth from DeFINES
            answer = mainstage.run(uid, candi_hardparams)
            for i in range(0, len(answer)):
                energy += answer[i][0].energy_total
                cycle += answer[i][0].latency_total1
            print("==> Current Energy & Latency: {:.6f} mJ, {:.6f} million cycles".format(energy * 1e-9, cycle * 1e-6)) #change energy's unit from pJ to mJ
        # energy = -(energy + random.randint(0, 1000))
        # cycle = -(cycle + random.randint(0, 1000))
        if self.Hardware_Arch == "ScaleSim" or self.Hardware_Arch == "DL2":
            energy = -(energy)
            cycle = -(cycle)
        return network_score, energy, cycle

    def _generate_initial_data(self, image, n):
        with torch.no_grad():
            print("==> Generate initial data for optimization..")
            train_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds, n=n, q=1).squeeze(1))
            # generate training data
            if self.benchmark == "sss":
                train_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1)) # scaling factor
                train_x = torch.cat((train_x_sf, train_x_hd), 1)
                train_obj = []

                for candidate in tqdm(train_x.tolist()):
                    candidate = list(map(int, candidate))
                    arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3], candidate[4])
                    accelerator = candidate[5:]
                    network_score, energy, cycle = self._rbflex_and_dse(image, arch, accelerator)

                    if self.Hardware_Arch == "DL2":
                        train_obj.append([network_score, energy, cycle])
                    elif self.Hardware_Arch == "ScaleSim":
                        train_obj.append([network_score, cycle])
                    elif self.Hardware_Arch == "DeFiNES":
                        train_obj.append([network_score, energy, cycle])
                train_obj = torch.tensor(train_obj, **tkwargs)
            elif self.benchmark == "201":
                train_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1))
                train_x = torch.cat((train_x_sf, train_x_hd), 1)
                train_obj = []
                for candidate in tqdm(train_x.tolist()):
                    candidate = list(map(int, candidate))
                    action = candidate[0:6]
                    arch = self.policy.generate_arch(action)
                    accelerator = candidate[6:]
                    network_score, energy, cycle = self._rbflex_and_dse(image, arch, accelerator)
                    if self.Hardware_Arch == "DL2":
                        train_obj.append([network_score, energy, cycle])
                    elif self.Hardware_Arch == "ScaleSim":
                        train_obj.append([network_score, cycle])
                    elif self.Hardware_Arch == "DeFiNES":
                        train_obj.append([network_score, energy, cycle])
                train_obj = torch.tensor(train_obj, **tkwargs)
        return train_x, train_obj

    def _initialize_model(self, train_x, train_obj, bounds):
        # define models for objective and constraint
        train_x = train_x.to(**tkwargs)
        train_xn = normalize(train_x, bounds)
        models = []
        train_obj = train_obj.to(**tkwargs)
        train_obj_stand = standardize(train_obj)
        for i in range(train_obj_stand.shape[-1]):
            train_y = train_obj_stand[..., i:i + 1]
            models.append(
                SingleTaskGP(train_xn, train_y)  # Gaussian Priocess
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    """[Todo]Check this program."""

    def _get_new_data(self, image, acqu_algo, model, train_x, sampler):
        AF_flag = True
        if acqu_algo == "qNEHVI":
            candidates = self._optimize_qnehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qEHVI":
            candidates = self._optimize_qehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qNParEGO":
            candidates = self._optimize_qnparego_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "random":
            AF_flag = False
            new_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds, n=self.BATCH_SIZE, q=1).squeeze(1))
            if self.benchmark == "sss":
                new_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)
            elif self.benchmark == "201":
                new_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds, n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)

        else:
            print("Select correct acquation function from [qNEHVI, qEHVI, qNParEGO, random]")
            exit()

        if AF_flag:
            new_x = torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_fornorm))

            if self.benchmark == "sss":
                sf_new_x = new_x[:, :5]
                sf_new_x = torch.round(sf_new_x / 8) * 8
                new_x[:, :5] = sf_new_x

        with torch.no_grad():
            new_obj = []
            for candidate in new_x.tolist():
                if self.benchmark == "sss":
                    candidate = list(map(int, candidate))
                    arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3], candidate[4])
                    accelerator = candidate[5:]
                elif self.benchmark == "201":
                    candidate = list(map(int, candidate))
                    action = candidate[0:6]
                    arch = self.policy.generate_arch(action)
                    accelerator = candidate[6:]
                network_score, energy, cycle = self._rbflex_and_dse(image, arch, accelerator)
                if self.Hardware_Arch == "DL2":
                    new_obj.append([network_score, energy, cycle])
                elif self.Hardware_Arch == "ScaleSim":
                    new_obj.append([network_score, cycle])
                elif self.Hardware_Arch == "DeFiNES":
                    new_obj.append([network_score, energy, cycle])
            new_obj = torch.tensor(new_obj, **tkwargs)
        return new_x, new_obj

    """[Todo] Solve the error. Float for the destination and Double for the source """

    def _optimize_qnehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            X_baseline=normalize(train_x, self.bounds_fornorm),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        return candidates

    def _optimize_qehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, self.bounds_fornorm)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point,
            Y=pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return candidates

    def _optimize_qnparego_and_get_observation(self, model, train_x, sampler):
        train_x = normalize(train_x, self.bounds_fornorm)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(self.mobo_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds_forstard,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates

    """[Todo] change negative to positive"""

    def plots(self, hsv_list, train_obj):
        fig = plt.figure()
        ax_re = fig.add_subplot(1, 2, 1, projection='3d')
        train_obj = train_obj.cpu().numpy()
        ax_re.scatter(
            train_obj[:, 0], -1 * train_obj[:, 1], -1 * train_obj[:, 2], alpha=0.8,
        )
        ax_re.set_title("AF: {} H-Volume: {}".format(self.acqu_algo, hsv_list[-1]))
        ax_re.set_xlabel("network score")
        ax_re.set_ylabel("Energey")
        ax_re.set_zlabel("Cycle count")

        ax_2d = fig.add_subplot(1, 2, 2)
        ax_2d.plot(hsv_list)
        ax_2d.set_xlabel("Iteration")
        ax_2d.set_ylabel("H-volume")

        plt.show()

    def run(self):
        print()
        print('+++++++++++ Optimization +++++++++++')

        train_x, train_obj = self._generate_initial_data(self.image, n=self.n_init_size)

        mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
        # sys.exit()
        hvs = []
        # Reference points
        min_values, _ = torch.min(train_obj, dim=0)
        if self.Hardware_Arch == "DL2":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.ENERGY_IDX] = min_values[self.ENERGY_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        elif self.Hardware_Arch == "ScaleSim":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        elif self.Hardware_Arch == "DeFiNES":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.ENERGY_IDX] = min_values[self.ENERGY_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=train_obj)
        # DominatedPartitioning 类实例化了一个对象 bd，其中 ref_point 是参考点，Y 是一组训练目标值。
        # 参考点通常是一个标量向量，表示在所有目标上都不被支配的最差解。
        # train_obj 是多目标优化中的目标值矩阵
        volume = bd.compute_hypervolume().item()
        # 调用 compute_hypervolume 方法来计算超体积。
        # 计算出的超体积表示由参考点和目标值定义的空间的体积。
        # item() 方法用于将结果转换为一个 Python 标量
        hvs.append(volume)
        print("          [init] Hypervolume: {}".format(self.N_BATCH, hvs[-1]))

        for iteration in range(1, self.N_BATCH + 1):
            fit_gpytorch_mll(mll)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size(
                [self.MC_SAMPLES]))  # define the qEI and qNEI acquisition modules using a QMC sampler
            new_x, new_obj = self._get_new_data(self.image, self.acqu_algo, model, train_x, sampler)

            train_x = torch.cat([train_x, new_x])  # 将多个张量沿指定维度进行连接
            train_obj = torch.cat([train_obj, new_obj])

            bd = DominatedPartitioning(ref_point=self.ref_point, Y=train_obj)
            volume = bd.compute_hypervolume().item()
            hvs.append(volume)

            if not self.acqu_algo == "random":
                mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)

            print("iteration [{}/{}] Hypervolume: {}".format(iteration, self.N_BATCH, hvs[-1]))

        # Show top-5 result
        print()
        print('+++++++++++ Result +++++++++++')
        torch.set_printoptions(precision=2, linewidth=100)
        print("H-Volume: ", hvs[-1])
        if self.Hardware_Arch == "DL2":
            if self.benchmark == "sss":
                optimal_x = train_x[-1 - self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1 - self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]),
                                                   int(optimal_x[i, 3]), int(optimal_x[i, 4]))
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i, 0].item())
                    print("\tEnergy (FAKE): ", -1 * optimal_obj[i, 1].item())
                    print("\tCycle Count (FAKE): ", int(-1 * optimal_obj[i, 2].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1 - self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1 - self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i, 0].item())
                    print("\tEnergy (FAKE): ", -1 * optimal_obj[i, 1].item())
                    print("\tCycle Count (FAKE): ", int(-1 * optimal_obj[i, 2].item()))
        elif self.Hardware_Arch == "ScaleSim":
            if self.benchmark == "sss":
                optimal_x = train_x[-1 - self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1 - self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]),
                                                   int(optimal_x[i, 3]), int(optimal_x[i, 4]))
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tHardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i, 0].item())
                    print("\tCycle Count (FAKE): ", int(-1 * optimal_obj[i, 1].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1 - self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1 - self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i, 0].item())
                    print("\tCycle Count (FAKE): ", int(-1 * optimal_obj[i, 1].item()))
        elif self.Hardware_Arch == "DeFiNES":
            pass
            if self.benchmark == "sss":
                optimal_x = train_x[-1 - self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1 - self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]),
                                                   int(optimal_x[i, 3]), int(optimal_x[i, 4]))
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDeFiNES Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", (optimal_obj[i, 0].item()))
                    print("\tEnergy: ", (optimal_obj[i, 1].item()) / 1e9, "mj")
                    print("\tCycle Count: ", (int(optimal_obj[i, 2].item()) / 1e6), "milion_cycle")

        try:
            if self.Hardware_Arch == "ScaleSim":
                base_path = 'COFleX_result/ScaleSim_SSS_' + str(self.mapping)
                os.makedirs(base_path, exist_ok=True)
                save_path = 'COFleX_result/ScaleSim_SSS_' + str(self.mapping) + '/hvs.csv'
                np.savetxt(save_path, hvs)
                save_path = 'COFleX_result/ScaleSim_SSS_' + str(self.mapping) + '/train_output.csv'
                np.savetxt(save_path, train_obj.cpu().numpy())
                save_path = 'COFleX_result/ScaleSim_SSS_' + str(self.mapping) + '/train_input.csv'
                np.savetxt(save_path, train_x.cpu().numpy())
                self.plots(hvs, train_obj)
            elif self.Hardware_Arch == "DeFiNES":
                base_path = 'COFleX_result/DeFiNES_SSS_' + str(self.mapping)
                os.makedirs(base_path, exist_ok=True)
                save_path = 'COFleX_result/DeFiNES_SSS_' + str(self.mapping) + '/hvs.csv'
                np.savetxt(save_path, hvs)
                save_path = 'COFleX_result/DeFiNES_SSS_' + str(self.mapping) + '/train_output.csv'
                np.savetxt(save_path, train_obj.cpu().numpy())
                save_path = 'COFleX_result/DeFiNES_SSS_' + str(self.mapping) + '/train_input.csv'
                np.savetxt(save_path, train_x.cpu().numpy())
                self.plots(hvs, train_obj)
        except Exception:
            pass
