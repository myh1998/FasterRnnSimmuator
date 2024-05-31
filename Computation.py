import numpy as np
import Simulator.unit_values as unit
import Simulator.PE_Macro as PE
import Simulator.Buffers as Buffers
import Simulator.Energy_model as Energy
import Simulator.Cycle_model as Cycle
import torch.nn.functional as F
import time
import torch
import os
import re
from torch import Tensor
import random

FAKE = True

class DL2:

    def __init__(self, optimized_components, mapping="rw"):
        self.horizonal_PEunits = 0
        self.vertical_PEunits = 0
        self.horizonal_PEmacros = 0
        self.vertical_PEmacros = 0
        self.activation_buffer_size = 0
        self.weights_buffer_size = 0
        self.image_buffer_size = 0
        self.color_ch = {'Red':0, 'Green':1, 'Blue':2}
        self.CPU = CPU()
        self.BufferFeature = [Buffers.Buffer(),Buffers.Buffer()]
        self.BufferImage = Buffers.Buffer()
        self.residual = 0
        self.Bopt = False
        self.hardware_components_values = {"X1":0, "X2":0, "X3":30, "X4":30, "X5":30, "X6":30}
        self.optimized_comp = optimized_components
        self.mapping = mapping
        self.Energy_Model = Energy.Model()
        self.Cycle_Model = Cycle.Model()


    def run(self, image, layers_dist, Hard_Params, HW_test=False):

        ########################
        # Set hardware architecture
        ########################
        self.hardware_components_values["X1"] = Hard_Params[0]
        self.hardware_components_values["X2"] = Hard_Params[1]
        self.hardware_components_values["X3"] = Hard_Params[2]
        self.hardware_components_values["X4"] = Hard_Params[3]
        self.hardware_components_values["X5"] = Hard_Params[4]
        self.hardware_components_values["X6"] = Hard_Params[5]
        self.DL2 = DL2_COMP(self.hardware_components_values, mapping=self.mapping)


        ########################
        # Write image data on the image buffer
        ########################
        _, img_H, img_W, img_C = image.shape
        

        ########################
        # Run computing
        ########################
        buffer_number = 2
        
        list_layer = list(layers_dist.keys())
        layer_num = 0
        total_energy = 0
        total_cycles = 0
        total_latency = 0
        Layer_length = len(layers_dist)
        keys = list(layers_dist.keys())

        for ind, key in enumerate(layers_dist):

            if layers_dist[key] == {}:
                continue

            ########################
            # Convolution (input: 4D Numpy)
            ########################
            if layers_dist[key]['layer'] == 'convolution':

                ########################
                # Computation
                ########################
                output, PERFORMANCE = self.DL2.convolution(layers_dist[key]['param']['in_channel'],
                                              layers_dist[key]['param']['out_channel'],
                                              layers_dist[key]['param']['kernel_size'],
                                              layers_dist[key]['param']['padding'],
                                              layers_dist[key]['param']['stride'],
                                              np.array(layers_dist[key]['weights']),
                                              np.array(layers_dist[key]['bias']),
                                              layers_dist[key]['param']['input_size'],
                                              layers_dist[key]['param']['output_size'])
                
                ########################
                # Energy
                ########################
                #### [Computation]
                sub_energy_comp = self.Energy_Model.get_energy(PERFORMANCE["NUM_MULT"], 1, unit.UNIT_ENERGY_MULTI, "computation")
                sub_energy_comp += self.Energy_Model.get_energy(PERFORMANCE["NUM_ADD"], 1, unit.UNIT_ENERGY_ADDER, "computation")
                #### [Data movement] Some module -> PE
                if ind == 0:
                    #### weight buffer -> PE
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_WEIGHT"], unit.UNIT_ENERGY_WRITE, "datamovement")
                    #### image buffer -> PE
                    sub_energy_data_in += self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toPE_fmBUFIMG, "datamovement")
                else:
                    previous_layer =  layers_dist[keys[ind-1]]['layer']
                    #### [batchnorm] W: weight buffer -> PE  A: activation buffer -> PE
                    if previous_layer == "batchnorm":
                        sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_WEIGHT"], unit.UNIT_ENERGY_WRITE, "datamovement")
                        sub_energy_data_in += self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toPE_fmBUFACT, "datamovement")
                    #### [relu] W: weight buffer -> PE  A: ReLU -> PE
                    elif previous_layer == "relu":
                        sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_WEIGHT"], unit.UNIT_ENERGY_WRITE, "datamovement")
                        sub_energy_data_in += self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toPE_fmAdderReLU, "datamovement")
                    #### [maxpool] W: weight buffer -> PE  A: activation buffer -> PE
                    elif previous_layer == "maxpool":
                        sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_WEIGHT"], unit.UNIT_ENERGY_WRITE, "datamovement")
                        sub_energy_data_in += self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toPE_fmBUFACT, "datamovement")
                    else:
                        sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_WEIGHT"], unit.UNIT_ENERGY_WRITE, "datamovement")
                        sub_energy_data_in += self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toPE_fmBUFACT, "datamovement")

                #### [Data movement] PE -> buffer
                next_layer =  layers_dist[keys[ind+1]]['layer']
                sub_energy_data_out = 0
                if not next_layer == "relu" or next_layer == "maxpool":
                    sub_energy_data_out = self.Energy_Model.get_energy(1, PERFORMANCE["SIZE_OUTPUT"], unit.UNIT_ENERGY_DATA_toBUFACT_fmPE, "datamovement")
                sub_energy = sub_energy_comp + sub_energy_data_in + sub_energy_data_out
                
                ########################
                # Cycle
                ########################
                cycles_mult = PERFORMANCE["NUM_MULT"] * unit.UNIT_CYCLE_MULTI
                cycles_adder = PERFORMANCE["NUM_ADD"] * unit.UNIT_CYCLE_ADDER
                cycles_data_in = (PERFORMANCE["SIZE_W_WEIGHT"]+PERFORMANCE["SIZE_W_ACTIVATION"]) * unit.UNIT_CYCLE_WRITE
                cycles_data_out = 0
                if not next_layer == "relu" or next_layer == "maxpool":
                    cycles_data_out = (PERFORMANCE["SIZE_OUTPUT"]) * unit.UNIT_CYCLE_DATA_toBUFACT_fmPE
                sub_cycles = cycles_mult + cycles_adder + cycles_data_in + cycles_data_out

                ########################
                # Letency
                ########################
                sub_latency = sub_cycles * unit.UNIT_LATENCY

                total_energy += sub_energy
                total_cycles += sub_cycles
                total_latency += sub_latency

            ########################
            # Max Pooling (input: 4D Torch Tensor)
            ########################
            elif layers_dist[key]['layer'] == 'maxpool':
                input = torch.from_numpy(self.stored).clone()

                ########################
                # Computation
                ########################
                output, PERFORMANCE_MAXP = self.DL2.maxpool(layers_dist[key]['param']['kernel_size'],
                                          layers_dist[key]['param']['stride'],
                                          layers_dist[key]['param']['padding'],
                                          layers_dist[key]['param']['input_size'],
                                          layers_dist[key]['param']['output_size'])
                
                ########################
                # Energy
                ########################
                #### [Computation]
                sub_energy_comp = self.Energy_Model.get_energy(PERFORMANCE_MAXP["NUM_MAXP"], 1, unit.UNIT_ENERGY_MAXP, "computation")
                #### [Data movement] Some module -> Maxpool
                previous_layer =  layers_dist[keys[ind-1]]['layer']
                if previous_layer == "convolution":
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_MAXP["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toMAXP_fmPE, "datamovement")
                elif previous_layer == "relu":
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_MAXP["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toMAXP_fmAdderReLU, "datamovement")
                elif previous_layer == "batchnorm":
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_MAXP["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toMAXP_fmBUFACT, "datamovement")
                else:
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_MAXP["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toMAXP_fmBUFACT, "datamovement")
                #### [Data movement] Maxpool -> buffer
                next_layer =  layers_dist[keys[ind+1]]['layer']
                sub_energy_data_out = 0
                if not next_layer == "relu":
                    sub_energy_data_out = self.Energy_Model.get_energy(1, PERFORMANCE_MAXP["SIZE_OUTPUT"], unit.UNIT_ENERGY_DATA_toBUFACT_fmMAXP, "datamovement")
                sub_energy = sub_energy_comp + sub_energy_data_in + sub_energy_data_out

                ########################
                # Cycle
                ########################
                cycles_comp = PERFORMANCE_MAXP["NUM_MAXP"] * unit.UNIT_CYCLE_MAXP
                cycles_data_in = PERFORMANCE_MAXP["SIZE_W_ACTIVATION"] * unit.UNIT_CYCLE_WRITE
                cycles_data_out = 0
                if not next_layer == "relu":
                    cycles_data_out = (PERFORMANCE_MAXP["SIZE_OUTPUT"]) * unit.UNIT_CYCLE_DATA_toBUFACT_fmMAXP
                sub_cycles = cycles_comp + cycles_data_in + cycles_data_out

                ########################
                # Letency
                ########################
                sub_latency = sub_cycles * unit.UNIT_LATENCY

                total_energy += sub_energy
                total_cycles += sub_cycles
                total_latency += sub_latency


                ########################
                # move next layer computed by DL2 or write output feature on buffer
                ########################
                if ind+1 < Layer_length:
                    next_layer = ind+1
                    if layers_dist[list_layer[next_layer]]['layer'] == 'batchnorm':
                        self.stored = output
                    elif layers_dist[list_layer[next_layer]]['layer'] == 'relu':
                        self.stored = output
                    else:
                        self.BufferFeature[buffer_number%2].write(output)
                        buffer_number += 1


            ########################
            # Batch Normalization (input: 4D Torch Tensor)
            ########################
            elif layers_dist[key]['layer'] == 'batchnorm':


                ########################
                # Computation
                ########################
                output, PERFORMANCE_BN = self.DL2.batchnorm(layers_dist[key]['mean'],
                                            layers_dist[key]['var'],
                                            layers_dist[key]['weights'],
                                            layers_dist[key]['bias'],
                                            layers_dist[key]['param']['input_size'],
                                            layers_dist[key]['param']['output_size'])
                
                ########################
                # Energy
                ########################
                #### [Computation]
                sub_energy_comp = self.Energy_Model.get_energy(PERFORMANCE_BN["NUM_BN"], 1, unit.UNIT_ENERGY_BATCHNORM, "computation")
                #### [Data movement] Some module -> BN
                sub_energy_data_in = 0
                #### [Data movement] BN -> buffer
                sub_energy_data_out = self.Energy_Model.get_energy(1, PERFORMANCE_BN["SIZE_OUTPUT"], unit.UNIT_ENERGY_DATA_toBUFACT_fmBATCHNORM, "datamovement")
                sub_energy = sub_energy_comp + sub_energy_data_in + sub_energy_data_out

                ########################
                # Cycle
                ########################
                cycles_comp = PERFORMANCE_BN["NUM_BN"] * unit.UNIT_CYCLE_BATCHNORM
                cycles_data_in = PERFORMANCE_BN["SIZE_W_ACTIVATION"] * unit.UNIT_CYCLE_WRITE
                cycles_data_out = (PERFORMANCE_BN["SIZE_OUTPUT"]) * unit.UNIT_CYCLE_DATA_toBUFACT_fmBATCHNORM
                sub_cycles = cycles_comp + cycles_data_in + cycles_data_out

                ########################
                # Letency
                ########################
                sub_latency = sub_cycles * unit.UNIT_LATENCY

                total_energy += sub_energy
                total_cycles += sub_cycles
                total_latency += sub_latency
                

                ########################
                # move next layer computed by DL2 or write output feature on buffer
                ########################
                if ind+1 < Layer_length:
                    next_layer = ind+1
                    # move next layer computed by DL2
                    if layers_dist[list_layer[next_layer]]['layer'] == 'maxpool':
                        self.stored = output
                    elif layers_dist[list_layer[next_layer]]['layer'] == 'relu':
                        self.stored = output
                    else:
                        self.BufferFeature[buffer_number%2].write(output)
                        buffer_number += 1


            ########################
            # ReLU (input: 4D Torch Tensor)
            ########################
            elif layers_dist[key]['layer'] == 'relu':

                ########################
                # Computation
                ########################
                output, PERFORMANCE_RELU = self.DL2.relu(layers_dist[key]['param']['input_size'],
                                            layers_dist[key]['param']['output_size'])

                ########################
                # Energy
                ########################
                #### [Computation]
                sub_energy_comp = self.Energy_Model.get_energy(PERFORMANCE_RELU["NUM_RELU"], 1, unit.UNIT_ENERGY_RELU, "computation")
                #### [Data movement] Some module -> ReLU
                previous_layer =  layers_dist[keys[ind-1]]['layer']
                if previous_layer == "convolution":
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_RELU["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toAdderReLU_fmPE, "datamovement")
                elif previous_layer == "maxpool":
                    sub_energy_data_in = self.Energy_Model.get_energy(1, PERFORMANCE_RELU["SIZE_W_ACTIVATION"], unit.UNIT_ENERGY_DATA_toAdderReLU_fmMAXP, "datamovement")
                #### [Data movement] ReLU -> buffer
                next_layer =  layers_dist[keys[ind+1]]['layer']
                sub_energy_data_out = 0
                if not next_layer == "maxpool" or next_layer == "convolution":
                    sub_energy_data_out = self.Energy_Model.get_energy(1, PERFORMANCE_RELU["SIZE_OUTPUT"], unit.UNIT_ENERGY_DATA_toBUFACT_fmAdderReLU, "datamovement")
                sub_energy = sub_energy_comp + sub_energy_data_in + sub_energy_data_out


                ########################
                # Cycle
                ########################
                cycles_comp = PERFORMANCE_RELU["NUM_RELU"] * unit.UNIT_CYCLE_RELU
                cycles_data_in = PERFORMANCE_RELU["SIZE_W_ACTIVATION"] * unit.UNIT_CYCLE_WRITE
                cycles_data_out = 0
                if not next_layer == "maxpool" or next_layer == "convolution":
                    cycles_data_out = (PERFORMANCE_RELU["SIZE_OUTPUT"]) * unit.UNIT_CYCLE_DATA_toBUFACT_fmAdderReLU
                sub_cycles = cycles_comp + cycles_data_in + cycles_data_out

                ########################
                # Letency
                ########################
                sub_latency = sub_cycles * unit.UNIT_LATENCY

                total_energy += sub_energy
                total_cycles += sub_cycles
                total_latency += sub_latency

                ########################
                # write output feature on buffer
                ########################
                self.BufferFeature[buffer_number%2].write(output)
                buffer_number += 1


            ########################
            # Average Pooling
            ########################
            elif layers_dist[key]['layer'] == 'avepool':
                input = self.BufferFeature[(buffer_number-1)%2].read2CPU()
                input = torch.from_numpy(input).clone()

                ########################
                # Computation
                ########################
                output = self.CPU.avepool(input.float(),
                                          layers_dist[key]['param']['kernel_size'],
                                          layers_dist[key]['param']['stride'],
                                          layers_dist[key]['param']['padding'])

                ########################
                # write output feature on buffer
                ########################
                self.BufferFeature[buffer_number%2].writefromCPU(output)
                buffer_number += 1

            ########################
            # Fully Connected
            ########################
            elif layers_dist[key]['layer'] == 'fc':
                input = self.BufferFeature[(buffer_number-1)%2].read2CPU()
                input = torch.from_numpy(input).clone()

                ########################
                # Computation
                ########################
                """[Todo] write a code for fc."""
                output = output

        return total_energy, total_cycles, total_latency


"""
Computation for DL2 Accelerator
Convolution and Fully Connected
"""
class DL2_COMP(DL2):

    def __init__(self, hardware_components_values, mapping="rw"):
        # set params
        self.horizonal_PEmacros = hardware_components_values["X1"]
        self.vertical_PEmacros = hardware_components_values["X2"]
        self.horizonal_PEunits = hardware_components_values["X3"]
        self.vertical_PEunits = hardware_components_values["X4"]


        # PE Unit Array and PE Macro Array for each PE Unit & Data memory
        self.PE_Macro_Array = [[PE.PE_Macro()]*self.horizonal_PEmacros]*self.vertical_PEmacros
        self.PE_Unit_Array = [[self.PE_Macro_Array]*self.horizonal_PEunits]*self.vertical_PEunits
        self.PE_Unit_Array = np.array(self.PE_Unit_Array)
        self.mapping = mapping

    ########################
    # Convolution
    ########################
    def convolution(self, in_ch, fm_ch, kernel_size, padding, stride, weights, bias, input_size, output_size):
        
        input = np.zeros(input_size)
        output = np.zeros(output_size)
        input = np.pad(input, ((0,0),(0,0),(padding,padding),(padding,padding)), 'constant')
        _, C_IF, H_IF, W_IF = input.shape
        num_rows_out = int(np.floor((H_IF-kernel_size)/stride)+1)
        num_col_out = int(np.floor((W_IF-kernel_size)/stride)+1)
        n_ifch_perPEunit = self.vertical_PEmacros/kernel_size
        n_PEunit_perfmch = int(np.ceil(in_ch/n_ifch_perPEunit)) # How many PEunit for one output channel
        n_fmch_periter = self.vertical_PEunits/n_PEunit_perfmch # How many output channel can be computed by one iteration
        n_iter_forfmch = np.ceil(1/n_fmch_periter)
        n_iter_foroutrow = np.ceil(num_rows_out/(self.horizonal_PEunits*self.horizonal_PEmacros)) # How many iteration is need to compute outrow
        n_iter_forweights = int(np.ceil(fm_ch/n_fmch_periter))

        q = num_rows_out
        Ln_outrow_periter = []
        while q >= self.horizonal_PEunits*self.horizonal_PEmacros:
            Ln_outrow_periter.append(self.horizonal_PEunits*self.horizonal_PEmacros)
            q -= self.horizonal_PEunits*self.horizonal_PEmacros
        if not q <= 0:
            Ln_outrow_periter.append(q)

        q = n_PEunit_perfmch
        Ln_PEunit_perfmch = []
        if q > self.vertical_PEunits:
            while q >= self.vertical_PEunits:
                Ln_PEunit_perfmch.append(self.vertical_PEunits)
                q -= self.vertical_PEunits
            if not q <= 0:
                Ln_PEunit_perfmch.append(q)
        else:
            if self.vertical_PEunits/n_PEunit_perfmch > 2:
                Ln_PEunit_perfmch.append(self.vertical_PEunits)
            else:
                Ln_PEunit_perfmch.append(n_PEunit_perfmch)

        n_used_PEmacro = []
        q = in_ch*kernel_size
        if q > self.vertical_PEmacros:
            while q >= self.vertical_PEmacros:
                n_used_PEmacro.append(self.vertical_PEmacros)
                q -= self.vertical_PEmacros
            if not q <= 0:
                n_used_PEmacro.append(q)
        else:
            n_used_PEmacro.append(q)

        
        ##################################################################
        #   Mapping: Weight Stationary (Priority to Output Channel)
        #   To maximize convolution reuse and filter reuse
        ##################################################################
        
        size_write_weight = 0
        size_write_activation = 0
        size_write_weight_rs = 0
        size_write_activation_rs = 0
        counter_mult = 0
        counter_adder = 0

        for iter_w in range(n_iter_forweights):
            loaded_input = np.zeros((C_IF, H_IF))
            no_w = int((iter_w/n_iter_forfmch)*np.ceil(n_fmch_periter))
            for cols_idx, cols_PEunit in enumerate(Ln_PEunit_perfmch):
                n = 0
                for PEunit_v in range(cols_PEunit):
                    NO_W = int(no_w+(PEunit_v/n_PEunit_perfmch))
                    upe_idx = cols_idx*self.vertical_PEunits+PEunit_v
                    CH_OUT = NO_W
                    if NO_W >= fm_ch:
                        break
                    for macro_v in range(n_used_PEmacro[int(upe_idx%n_PEunit_perfmch)]):
                        KS_W = int(n%kernel_size)
                        CH_W = int(n/kernel_size%in_ch)

                        ########################
                        # set weights
                        ########################
                        sub_w = weights[NO_W,CH_W,KS_W,:]
                        _,W = sub_w.reshape(1,-1).shape
                        if self.mapping == "ws":
                            size_write_weight += W
                        elif self.mapping == "rs":
                            size_write_weight_rs += W

                        CH_IN = CH_W
                        ########################
                        # set input
                        ########################
                        for rows_idx, rows_PEmacro in enumerate(Ln_outrow_periter):
                            if rows_idx==0:
                                d_in = 0
                            else:
                                d_in += Ln_outrow_periter[rows_idx-1]
                            for row_PEmacro in range(rows_PEmacro):
                                H_IN = (d_in+row_PEmacro)*stride+KS_W
                                sub_in = input[0,CH_IN,H_IN,:]
                                _,A = sub_in.reshape(1,-1).shape
                                size_write_activation += A
                                if loaded_input[CH_IN,H_IN] == False:
                                    loaded_input[CH_IN,H_IN] = True
                                    size_write_activation_rs += A

                        ########################
                        # Multiplation
                        ########################
                        for rows_idx, rows_PEmacro in enumerate(Ln_outrow_periter):
                            if rows_idx==0:
                                d_in = 0
                            else:
                                d_in += Ln_outrow_periter[rows_idx-1]
                            for row_PEmacro in range(rows_PEmacro):
                                counter_mult += 1

                        ########################
                        # Adder
                        ########################
                        for rows_idx, rows_PEmacro in enumerate(Ln_outrow_periter):
                            if rows_idx==0:
                                d_in = 0
                            else:
                                d_in += Ln_outrow_periter[rows_idx-1]
                            for row_PEmacro in range(rows_PEmacro):
                                counter_adder += 1
                        n+=1
                if NO_W >= fm_ch:
                    break


        if self.mapping == "ws":
            NUM_MULT = counter_mult
            NUM_ADD = counter_adder
            SIZE_W_WEIGHT = size_write_weight
            SIZE_W_ACTIVATION = size_write_activation
            

        elif self.mapping == "rs":
            NUM_MULT = counter_mult
            NUM_ADD = counter_adder
            SIZE_W_WEIGHT = size_write_weight_rs
            SIZE_W_ACTIVATION = size_write_activation_rs

        
        _,OU = output.reshape(1,-1).shape
        SIZE_OUTPUT = OU
        PERFORMANCE = {"NUM_MULT": NUM_MULT, "NUM_ADD": NUM_ADD, "SIZE_W_WEIGHT": SIZE_W_WEIGHT*unit.BITS_WEIGHTS, "SIZE_W_ACTIVATION": SIZE_W_ACTIVATION*unit.BITS_ACTIVATION, "SIZE_OUTPUT": SIZE_OUTPUT*unit.BITS_ACTIVATION}

        return output, PERFORMANCE

    ########################
    # Maxpooling
    ########################
    def maxpool(self, kernel_size, stride, padding, input_size, output_size):
        input = np.zeros(input_size)
        #output = F.max_pool2d(input=input, kernel_size=kernel_size, stride=stride, padding=padding)
        output = np.zeros(output_size)

        _,A = input.reshape(1,-1).shape
        SIZE_W_ACTIVATION = A
        _,OU = output.reshape(1,-1).shape
        SIZE_OUTPUT = OU
        NUM_MAXP = 1

        PERFORMANCE = {"NUM_MAXP": NUM_MAXP, "SIZE_W_ACTIVATION": SIZE_W_ACTIVATION*unit.BITS_ACTIVATION, "SIZE_OUTPUT": SIZE_OUTPUT*unit.BITS_ACTIVATION}

        return output, PERFORMANCE

    ########################
    # Batch Normalization
    ########################
    def batchnorm(self, mean, var, weight, bias, input_size, output_size):
        input = np.zeros(input_size)
        #output = F.batch_norm(input=input, running_mean=mean, running_var=var, weight=weight, bias=bias)
        output = np.zeros(output_size)

        _,A = input.reshape(1,-1).shape
        SIZE_W_ACTIVATION = A
        _,OU = output.reshape(1,-1).shape
        SIZE_OUTPUT = OU
        NUM_BN = 1

        PERFORMANCE = {"NUM_BN": NUM_BN, "SIZE_W_ACTIVATION": SIZE_W_ACTIVATION*unit.BITS_ACTIVATION, "SIZE_OUTPUT": SIZE_OUTPUT*unit.BITS_ACTIVATION}

        return output, PERFORMANCE

    ########################
    # ReLU activation
    ########################
    def relu(self, input_size, output_size):
        input = np.zeros(input_size)
        #output = F.relu(input=input)
        output = np.zeros(output_size)

        _,A = input.reshape(1,-1).shape
        SIZE_W_ACTIVATION = A
        _,OU = output.reshape(1,-1).shape
        SIZE_OUTPUT = OU
        NUM_RELU = 1

        PERFORMANCE = {"NUM_RELU": NUM_RELU, "SIZE_W_ACTIVATION": SIZE_W_ACTIVATION*unit.BITS_ACTIVATION, "SIZE_OUTPUT": SIZE_OUTPUT*unit.BITS_ACTIVATION}

        return output, PERFORMANCE


"""
Computation for CPU
Maxpooling, Average Pooling, Batch Normalization
"""
class CPU:

    def __init__(self):
        pass

    def avepool(self, input, kernel_size, stride, padding):
        out = F.avg_pool2d(input=input, kernel_size=kernel_size, stride=stride, padding=padding)
        return out
