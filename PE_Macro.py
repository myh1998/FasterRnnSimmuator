import numpy as np
from numpy.lib.stride_tricks import sliding_window_view



class PE_Macro:

    def __init__(self):
        self.weights = 0
        self.loaded_weights = False # flag for loading weights
        self.input = 0
        self.loaded_input = False # flag for loading input
        self.P = 0

    def write_Weights(self, subweights):
        self.weights = np.array(subweights)
        self.loaded_weights = True

    def write_Input(self, subinput):
        self.input = np.array(subinput)
        self.loaded_input = True

    def get_psum(self):
        return self.P

    def run(self, kernel_size, stride):
        if self.loaded_weights and self.loaded_input:
            """
            # Too slow (214s for one 7×7 convolution by M1 CPU) ######################
            for idx, v in enumerate(self.psum):
                st_idx = idx*stride
                ed_idx = st_idx+kernel_size
                self.psum[idx] = np.dot(self.input[st_idx:ed_idx],weights_t)
            """

            """
            30x faster (7s for one 7×7 convolution by M1 CPU) ######################
            """
            self.W = self.weights.T
            self.input_view = sliding_window_view(self.input,kernel_size)
            self.F = self.input_view[::stride,:]
            self.P = np.dot(self.F, self.W) # for example, p(1,1) = W1*F(1,1)+W2*F(1,2)+W3*F(1,3)
