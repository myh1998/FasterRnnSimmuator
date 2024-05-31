import numpy as np


class Buffer:

    def __init__(self, buffer_size=None):
        self.buffer_size = buffer_size
        self.buffer = 0
        self.num_write = 0
        self.num_read = 0
        self.num_read2CPU = 0
        self.num_writefromCPU = 0
        self.size_read = 0

    def define_buffer_size(self, buffer_size):
        self.buffer_size =  buffer_size

    def write(self, data):
        self.buffer = data
        self.num_write += 1

    def writefromCPU(self, data):
        self.buffer = data
        self.num_writefromCPU += 1

    def read(self):
        self.num_read += 1
        reshaped_data = self.buffer.reshape(1,-1)
        H,W = reshaped_data.shape
        self.size_read = W #bits
        return self.buffer

    def read2CPU(self):
        self.num_read2CPU += 1
        return self.buffer
