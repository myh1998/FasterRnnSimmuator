import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import re
from torchvision.datasets import ImageFolder
from design_space.models import get_cell_based_tiny_net
from nats_bench import create
import random
from tqdm import tqdm
import numexpr as ne
import time
import torchvision
import torchvision.transforms as transforms
import os
import json
import sys

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = "cpu"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class RBFleX:

    def __init__(self,
                 dataset, score_input_batch, img_root,
                 benchmark, benchmark_root,
                 n_hyper):

        self.device = torch.device(DEVICE)
        self.score_input_batch = score_input_batch
        self.benchmark = benchmark

        print()
        print('+++++++++++++++ Init RBFleX +++++++++++++++++')

        print('==> Reproducibility..')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)

        # Image Data
        print('==> Build dataloader..')
        if dataset == "ImageNet":
            norma = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                norma,
            ])
            imgset = ImageFolder(root=img_root, transform=train_transform)
            img_loader = torch.utils.data.DataLoader(imgset, batch_size=self.score_input_batch, shuffle=True,
                                                     num_workers=1, pin_memory=True)
            data_iterator = iter(img_loader)
            self.input_score, _ = next(data_iterator)
        elif dataset == 'cifar100':
            train_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])
            cifar100_training = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=True, download=True,
                                                              transform=train_transform)
            img_loader = torch.utils.data.DataLoader(cifar100_training, shuffle=True, num_workers=4,
                                                     batch_size=self.score_input_batch, pin_memory=True)
            data_iterator = iter(img_loader)
            self.input_score, _ = next(data_iterator)

        # Design Space
        print('==> Build design space..')
        if self.benchmark == "sss":
            print('Loading...NATS-Bench-SSS')
            self.searchspace = create(benchmark_root, 'sss', fast_mode=True, verbose=False)
        elif self.benchmark == "201":
            print('Loading...NAS-Bench-201')
            self.searchspace = create(benchmark_root, 'tss', fast_mode=True, verbose=False)

        # Hyperparameter Detection Algorithm
        print('==> Search the optimal hyperparameter for RBFleX..')
        N_GAMMA = n_hyper
        self.GAMMA_K, self.GAMMA_Q = self._hyperparameterDA(N_GAMMA)
        print('Hyperparameter\n\t[Activation: {}, Final input: {}]'.format(self.GAMMA_K, self.GAMMA_Q))

    def run(self, image, arch):
        uid = 0
        ########################
        # NAS
        ########################
        if self.benchmark == "sss" or self.benchmark == "201":
            uid = self.searchspace.query_index_by_arch(arch)
            config = self.searchspace.get_net_config(uid, 'cifar100')
            candi_network = get_cell_based_tiny_net(config, backbone=False)
            candi_network = candi_network.to(self.device)
            backbone = get_cell_based_tiny_net(config, backbone=True)

        self.setting_FRCN = 'Test Opt Network'

        ########################
        # Train Optimal Network
        ########################
        # torch.save(network.state_dict(), './optimal_network.pth')

        ########################
        # Score the candidate network
        ########################
        # print('==> Score the candidate network..')
        network_score = self._evaluator(candi_network)
        # print('uid: {} score: {}'.format(uid,network_score))
        if np.isinf(network_score):
            network_score = -10000

        ########################
        # Prepare to pass the opt network infor to AutoDSE
        # self.layer_dist
        ########################
        self.layers_dist = {}
        self.Dict_size = self._get_inout_size(backbone, image)
        self.Dict_counter = 1
        self._get_modelDist(self.Dict_size)
        self._get_trainedparam(backbone)
        converted_layers = self.converted_layers_info(self.layers_dist)

        # print(converted_layers)
        def format_dict(d, indent=4):
            formatted_str = "{\n"
            for k, v in d.items():
                v_str = json.dumps(v, cls=NumpyEncoder).replace('"', "'")
                v_str = v_str.replace('\n', '\n' + ' ' * (indent + 4))
                formatted_str += ' ' * indent + f"{k}: {v_str},\n"
            formatted_str = formatted_str.rstrip(',\n') + '\n' + ' ' * (indent - 4) + '}'
            return formatted_str

        # if self.score_input_batch ==
        formatted_converted_layers = 'workload = ' + format_dict(converted_layers, indent=4)
        # print("==> Writing..")
        output_dir = os.path.join('inputs/WL/Meta_prototype/', str(uid))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, 'workload_faster_rcnn.py')
        with open(output_file_path, 'w') as file:
            file.write(formatted_converted_layers)
        # print(f'==> Converted current layers have been saved to {output_file_path}')
        # print(f'==> Converted current layers have been saved to {output_file_path}', end='\r')
        # sys.exit()
        return network_score, backbone, self.layers_dist, uid

    def converted_layers_info(self, layers_dist):
        layer_dist = layers_dist
        workload = {}
        default_operand_precision = {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}
        default_equation_relations = ['ix=1*ox+1*fx', 'iy=1*oy+1*fy']
        default_input_layer = {'equation': 'input', 'loop_dim_size': {'B': 1, 'K': 1, 'OY': 32, 'OX': 32},
                               'precision': 8, 'core_allocation': 1, 'memory_operand_links': {'O': 'I1'}}
        workload[-1] = default_input_layer
        pre_idx = -1
        for idx, (layer_name, layer_info) in enumerate(layer_dist.items()):
            if layer_info['layer'] == 'convolution':
                params = layer_info['param']
                default_memory_operand_links = {'O': 'O', 'W': 'I2', 'I': 'I1'}
                default_convolution_spatial_mapping = {'D1': ('K', 24), 'D3': ('OX', 1), 'D4': ('OY', 1)}
                converted_layer = {
                    'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
                    'equation_relations': default_equation_relations,
                    'loop_dim_size': {'B': 1, 'K': params['out_channel'], 'C': params['in_channel'],
                                      'OY': params['output_size'][2], 'OX': params['output_size'][3],
                                      'FY': params['kernel_size'], 'FX': params['kernel_size']},
                    'operand_precision': default_operand_precision,
                    'operand_source': {'W': [], 'I': [pre_idx]},
                    'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
                    # K is for output channel; C is for input channel; OX and OY are feature map spatial dimensions; FX and FY are weight spatial dimensions.
                    'constant_operands': ['W'],
                    'core_allocation': 1,
                    'spatial_mapping': default_convolution_spatial_mapping,
                    'memory_operand_links': default_memory_operand_links
                }
                workload[idx] = converted_layer
                pre_idx = idx
            elif layer_info['layer'] == 'batchnorm':
                # params = layer_info['param']
                # default_batchnorm_spatial_mapping = {'D1': ('C', 48), 'D3': ('OX', 1), 'D4': ('OY', 1)}
                # default_batchnorm_memory_operand_links = {'O': 'O','I': 'I1','mean': 'mean','var': 'var','weight': 'weight','bias': 'bias'}
                # default_batchnorm_operand_precision = {'O': 16, 'O_final': 8, 'W': 8, 'I': 8, 'mean': 8, 'var': 8, 'weight': 8, 'bias': 8, '(': 0, ')': 0}
                # converted_layer = {
                #     'equation': 'O[b][c][oy][ox]=(I[b][c][oy][ox]-mean[c])/sqrt(var[c])*weight[c]+bias[c]',
                #     'loop_dim_size': {'B': 1, 'C': 48, 'OY': params['output_size'][2], 'OX': params['output_size'][3]},
                #     'operand_precision': default_batchnorm_operand_precision,
                #     'operand_source': {'I': [pre_idx]},
                #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'C', 'OX': 'OX', 'OY': 'OY'}},
                #     'constant_operands': ['mean', 'var', 'weight', 'bias'],
                #     'core_allocation': 1,
                #     'spatial_mapping': default_batchnorm_spatial_mapping,
                #     'memory_operand_links': default_batchnorm_memory_operand_links
                # }
                # workload[idx] = converted_layer
                # pre_idx = idx
                pass
            elif layer_info['layer'] == 'relu':
                # params = layer_info['param']
                # default_relu_spatial_mapping = {'D1': ('C', 48), 'D3': ('OX', 1), 'D4': ('OY', 1)}
                # default_relu_memory_operand_links = {'O': 'O','I': 'I1'}
                # default_relu_operand_precision = {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}
                # converted_layer = {
                # 'equation': 'O[b][k][oy][ox]=ReLU(I[b][k][oy][ox])',
                # 'loop_dim_size': {
                #     'B': 1, 
                #     'K': params['input_size'][1], 
                #     'C': params['input_size'][1], 
                #     'OY': params['output_size'][2], 
                #     'OX': params['output_size'][3], 
                # },
                # 'operand_precision': default_relu_operand_precision,
                # 'operand_source': {'I': [pre_idx]},
                # 'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'K': 'K'}}, 
                # 'constant_operands': [],
                # 'core_allocation': 1,
                # 'spatial_mapping': default_relu_spatial_mapping, 
                # 'memory_operand_links': default_relu_memory_operand_links
                # }
                # workload[idx] = converted_layer
                # pre_idx = idx
                pass
            elif layer_info['layer'] == 'maxpool':
                # params = layer_info['param']
                # default_spatial_maxpool_mapping = {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)}
                # default_G = 64
                # converted_layer = {
                #     # 'equation': 'O[b][k][oy][ox] = max(I[b][k][oy*stride+fy][ox*stride+fx])',
                #     'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
                #     'loop_dim_size': {'B': 1, 'G': default_G, 'OY': params['output_size'][2], 'OX': params['output_size'][3], 'FY': params['kernel_size'], 'FX': params['kernel_size'], 'IX': 32, 'IY': 32},
                #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 0},
                #     'operand_source': {'I': [pre_idx]},
                #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
                #     'constant_operands': ['I'],
                #     'core_allocation': 1,
                #     'spatial_mapping': default_spatial_maxpool_mapping,
                #     'memory_operand_links': default_memory_operand_links
                # }
                # workload[idx] = converted_layer
                # pre_idx = idx
                pass
            else:
                raise ValueError(f"Unsupported layer type: {layer_info['layer']}")
        return workload

    def _get_inout_size(self, backbone, input):
        def hook(module, inp, out):
            with torch.no_grad():
                if isinstance(inp, tuple):
                    inp = inp[0]
                if isinstance(out, tuple):
                    out = out[0]
                name = str(backbone.count) + ":" + str(module)
                backbone.dict_size[name] = {}
                backbone.dict_size[name]['input_size'] = inp.shape
                backbone.dict_size[name]['output_size'] = out.shape
                backbone.count += 1

        self._hook_layers(backbone, hook)

        backbone.dict_size = {}
        backbone.count = 1
        backbone(input)
        return backbone.dict_size

    def _hook_layers(self, module, hook, depth=0):
        if len(list(module.named_children())) == 0:
            m = str(module)
            if any(layer_type in m for layer_type in
                   ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'Linear']):
                module.register_forward_hook(hook)

        for child_name, child_module in module.named_children():
            self._hook_layers(child_module, hook, depth + 1)

    def _get_trainedparam(self, backbone):
        key_opt = list(backbone.state_dict().keys())
        dist_opt = backbone.state_dict()
        countN = 0

        for key, value in self.layers_dist.items():
            if value['layer'] == 'convolution':
                self.layers_dist[key]['weights'] = dist_opt[key_opt[countN]]
                countN += 1
                if value['param']['bias']:
                    self.layers_dist[key]['bias'] = dist_opt[key_opt[countN]]
                    countN += 1
                else:
                    self.layers_dist[key]['bias'] = {}

            elif value['layer'] == 'batchnorm':
                self.layers_dist[key]['weights'] = dist_opt[key_opt[countN]]
                countN += 1
                self.layers_dist[key]['bias'] = dist_opt[key_opt[countN]]
                countN += 1
                self.layers_dist[key]['mean'] = dist_opt[key_opt[countN]]
                countN += 1
                self.layers_dist[key]['var'] = dist_opt[key_opt[countN]]
                countN += 1
                self.layers_dist[key]['num_batches_tracked'] = dist_opt[key_opt[countN]]
                countN += 1

            elif value['layer'] == 'fc':
                self.layers_dist[key]['weights'] = dist_opt[key_opt[countN]]
                countN += 1
                if value['param']['bias']:
                    self.layers_dist[key]['bias'] = dist_opt[key_opt[countN]]
                    countN += 1
                else:
                    self.layers_dist[key]['bias'] = {}

        if not countN == len(key_opt):
            print(countN)
            print(len(key_opt))
            print('[ERROR]: NAS.py Pre-trained Saving API did not work. Check program.')
            exit()

    def _get_modelDist(self, Dict_size):
        for key in Dict_size:
            m = re.split('[(,=)]', str(key))
            name = str(m[0])
            self.layers_dist[name] = {}
            if 'Conv2d' in name:
                self.layers_dist[name]['layer'] = 'convolution'
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['in_channel'] = int(m[1])
                self.layers_dist[name]['param']['out_channel'] = int(m[2])
                self.layers_dist[name]['param']['kernel_size'] = int(m[5])
                self.layers_dist[name]['param']['stride'] = int(m[10])
                if 'padding' in str(key):
                    self.layers_dist[name]['param']['padding'] = int(m[15])
                else:
                    self.layers_dist[name]['param']['padding'] = 0
                if m[-2] == 'True':
                    self.layers_dist[name]['param']['bias'] = True
                else:
                    self.layers_dist[name]['param']['bias'] = False
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

            elif 'BatchNorm2d' in name:
                self.layers_dist[name]['layer'] = 'batchnorm'
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['out_channel'] = int(m[1])
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

            elif 'ReLU' in name:
                self.layers_dist[name]['layer'] = 'relu'
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

            elif 'MaxPool2d' in name:
                self.layers_dist[name]['layer'] = 'maxpool'
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['kernel_size'] = int(m[2])
                self.layers_dist[name]['param']['stride'] = int(m[4])
                self.layers_dist[name]['param']['padding'] = int(m[6])
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

            # DL2 supports only maxpooling, so Avepooling is changed to maxpooling
            elif 'AvgPool2d' in name:
                self.layers_dist[name]['layer'] = 'maxpool'
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['kernel_size'] = int(m[2])
                self.layers_dist[name]['param']['stride'] = int(m[4])
                self.layers_dist[name]['param']['padding'] = int(m[6])
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

            elif 'Linear' in m:
                self.layers_dist[name]['layer'] = 'fc'
                m = re.split('[(,=)]', m)
                self.layers_dist[name]['param'] = {}
                self.layers_dist[name]['param']['in_channel'] = int(m[2])
                self.layers_dist[name]['param']['out_channel'] = int(m[4])
                if m[-2] == 'True':
                    self.layers_dist[name]['param']['bias'] = True
                else:
                    self.layers_dist[name]['param']['bias'] = False
                self.layers_dist[name]['param']['input_size'] = np.array(Dict_size[key]['input_size'])
                self.layers_dist[name]['param']['output_size'] = np.array(Dict_size[key]['output_size'])

    def normalize(self, x, axis=None):
        x_min = x.min(axis=axis, keepdims=True)
        x_max = x.max(axis=axis, keepdims=True)
        x_max[x_max == x_min] = 1
        x_min[x_max == x_min] = 0
        return (x - x_min) / (x_max - x_min)

    def _evaluator(self, network):

        def counting_forward_hook(module, inp, out):
            with torch.no_grad():
                arr = out.view(self.score_input_batch, -1)
                network.K = torch.cat((network.K, arr), 1)

        def counting_forward_hook_FC(module, inp, out):
            with torch.no_grad():
                if isinstance(inp, tuple):
                    inp = inp[0]
                network.Q = inp

        net_counter = list(network.named_modules())
        net_counter = len(net_counter)
        NC = 0
        for name, module in network.named_modules():
            NC += 1
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(counting_forward_hook)
            if NC == net_counter:
                module.register_forward_hook(counting_forward_hook_FC)

        with torch.no_grad():
            network.K = torch.empty(0, device=self.device)
            network.Q = torch.empty(0, device=self.device)
            network(self.input_score[0:self.score_input_batch, :, :, :].to(self.device))

            Output_matrix = network.K
            Last_matrix = network.Q
            Output_matrix = self.normalize(Output_matrix.cpu().numpy(), axis=0)
            Last_matrix = self.normalize(Last_matrix.cpu().numpy(), axis=0)

            # RBF kernel
            X_norm = np.sum(Output_matrix ** 2, axis=-1)
            K_Matrix = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A': X_norm[:, None],
                'B': X_norm[None, :],
                'C': np.dot(Output_matrix, Output_matrix.T),
                'g': self.GAMMA_K
            })
            Y_norm = np.sum(Last_matrix ** 2, axis=-1)
            Q_Matrix = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A': Y_norm[:, None],
                'B': Y_norm[None, :],
                'C': np.dot(Last_matrix, Last_matrix.T),
                'g': self.GAMMA_Q
            })

            # Score
            _, K = np.linalg.slogdet(K_Matrix)
            _, Q = np.linalg.slogdet(Q_Matrix)
            score = self.score_input_batch * (K + Q)

        return score

    def _hyperparameterDA(self, N_GAMMA):

        def counting_forward_hook(module, inp, out):
            with torch.no_grad():
                arr = out.view(self.score_input_batch, -1)
                network.K = torch.cat((network.K, arr), 1)

        def counting_forward_hook_FC(module, inp, out):
            with torch.no_grad():
                if isinstance(inp, tuple):
                    inp = inp[0]
                network.Q = inp

        GAMMA_K_list = []
        GAMMA_Q_list = []
        batch_space = random.sample(range(len(self.searchspace)), N_GAMMA)
        for id in tqdm(range(N_GAMMA)):
            uid = batch_space[id]
            config = self.searchspace.get_net_config(uid, 'cifar100')
            network = get_cell_based_tiny_net(config)
            network = network.to(self.device)
            net_counter = list(network.named_modules())
            net_counter = len(net_counter)
            NC = 0
            for name, module in network.named_modules():
                NC += 1
                if 'ReLU' in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                if NC == net_counter:
                    module.register_forward_hook(counting_forward_hook_FC)

            with torch.no_grad():
                network.K = torch.empty(0, device=self.device)
                network.Q = torch.empty(0, device=self.device)
                network(self.input_score[0:self.score_input_batch, :, :, :].to(self.device))
                Output_matrix = network.K
                Last_matrix = network.Q
                Output_matrix = Output_matrix.cpu().numpy()
                Last_matrix = Last_matrix.cpu().numpy()

            for i in range(self.score_input_batch - 1):
                for j in range(i + 1, self.score_input_batch):
                    z1 = Output_matrix[i, :]
                    z2 = Output_matrix[j, :]
                    m1 = np.mean(z1)
                    m2 = np.mean(z2)
                    M = (m1 - m2) ** 2
                    z1 = z1 - m1
                    z2 = z2 - m2
                    s1 = np.mean(z1 ** 2)
                    s2 = np.mean(z2 ** 2)
                    if s1 + s2 != 0:
                        candi_gamma_K = M / ((s1 + s2) * 2)
                        GAMMA_K_list.append(candi_gamma_K)

            for i in range(self.score_input_batch - 1):
                for j in range(i + 1, self.score_input_batch):
                    z1 = Last_matrix[i, :]
                    z2 = Last_matrix[j, :]
                    m1 = np.mean(z1)
                    m2 = np.mean(z2)
                    M = (m1 - m2) ** 2
                    z1 = z1 - m1
                    z2 = z2 - m2
                    s1 = np.mean(z1 ** 2)
                    s2 = np.mean(z2 ** 2)
                    if s1 + s2 != 0:
                        candi_gamma_Q = M / ((s1 + s2) * 2)
                        GAMMA_Q_list.append(candi_gamma_Q)

        GAMMA_K = np.min(np.array(GAMMA_K_list))  # Gamma for activation outputs
        GAMMA_Q = np.min(np.array(GAMMA_Q_list))  # Gamma for final input
        return GAMMA_K, GAMMA_Q
