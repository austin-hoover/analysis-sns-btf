import sys
import os
from os.path import join
from datetime import datetime
import h5py
import numpy as np


class H5Reader:
    def __init__(self, filename, folder='.'):
        self.path = join(folder, filename)
        self.filename = filename
        self.file = h5py.File(self.path, 'r')
        self.log = self.file['/log']
        self.config = self.file['/config']
        self.data = self.file['/scandata']
        self.timestamp = datetime.fromtimestamp(self.log[0, 'timestamp']).strftime("%m/%d/%Y, %H:%M:%S")
        
    def close(self):
        self.file.close()