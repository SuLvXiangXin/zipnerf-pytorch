import numpy as np

import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 
from inspect import getmembers, isfunction
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_path)

import abc

class Funcs: # support the current function calling format outside i.e. backend.funcs.fun_name(xx)
    def __init__(self, funcs):
        for func in funcs:
            self.__dict__[func.__name__] = func   #  i.e. self.func_name = func

class BaseBackend(metaclass=abc.ABCMeta):

    def __init__(self):
        backend_name = self._get_backend_name()
        self.funcs = Funcs(self._get_funcs(backend_name)) 
        # for func in self.funcs:
        #     self.__dict__[func.__name__] = func

        self.device = None

    @abc.abstractmethod  
    def synchronize(self):
        pass

    @abc.abstractmethod 
    def _get_backend_name(self):
        pass 


    def _get_funcs(self, backend_name):
        """get functions defined in different backend"""
        funcs = []
        for name,func in getmembers(backend_name):
            if not (str(name).startswith('__') and str(name).endswith('__')):
                funcs.append(func)
        return funcs

    def get_generator(self):
        return self.device.get_generator()


class DPCPPBackend(BaseBackend):
    def __init__(self):
        super().__init__()

        self.gpu_vendor = self.funcs.get_gpu_vendor()

        self.device=DeviceFactory.create_device(self.gpu_vendor)


    def _get_backend_name(self):  
        # TODO: auto compile
        # TODO: read function names from a file? or read from .so module automaticlly
        try:
            backend_name = __import__('extensions.dpcpp._dpcpp_backend',fromlist=['extensions.dpcpp'])
        except ImportError:
            # TODO self.compile()
            backend_name = __import__('extensions.dpcpp._dpcpp_backend',fromlist=['extensions.dpcpp'])
        return backend_name
        

    def synchronize(self):
        self.device.synchronize()

    def get_gpu_vendor(self):
        return self.gpu_vendor


class CUDABackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.gpu_vendor = 'nvidia'
        self.device = DeviceFactory.create_device(self.gpu_vendor)


    def _get_backend_name(self):
        funcs = []
        try:
            backend_name = __import__('_cuda_backend')
        except ImportError:
            # TODO:self.compile()
            backend_name = __import__('_cuda_backend')

        return backend_name

    def synchronize(self):
        self.device.synchronize()


class BaseDevice(metaclass=abc.ABCMeta):    
    @abc.abstractmethod 
    def synchronize(self):
        pass

    @abc.abstractmethod 
    def get_generator(self):
        pass

class IntelDevice(BaseDevice):
    def __init__(self) -> None:
        super().__init__()
        import intel_extension_for_pytorch
        self.synchronize = torch.xpu.synchronize
        torch.set_default_tensor_type ('torch.xpu.FloatTensor')

    def synchronize(self):
        self.synchronize()

    def get_generator(self):
        return torch.xpu.Generator()

class NvidiaDevice(BaseDevice):

    def synchronize(self):
        torch.cuda.synchronize() 

    def get_generator(self):
        return None


from threading import RLock
class DeviceFactory():
    device_name2class={'intel':IntelDevice,'nvidia':NvidiaDevice}
    @classmethod
    def create_device(cls,device_name):
        if device_name not in cls.device_name2class.keys():
            raise Exception('unsupported gpu device type...')
        return cls.device_name2class[device_name]()

class Backend:
    """
    Create and retain 1 backend instance 
    """
    # lazy singleton & only create a instance of a specific backend class when it is used
    backend_name2class = {'dpcpp':DPCPPBackend,'cuda':CUDABackend}
    backend=None
    name=None
    single_lock = RLock()

    @classmethod
    def set_backend(cls, backend_name):
        with cls.single_lock:
            if cls.backend is not None:
                raise Exception('Backend cannot be set repeatedly')

            if backend_name not in cls.backend_name2class.keys():
                raise Exception(str(f"Unsupported backend type:{backend_name}. Currently [{cls.backend_name2class.keys()}] is supported."))
            cls.backend=cls.backend_name2class[backend_name]()
            cls.name = backend_name


    @classmethod
    def get_backend(cls):
        if cls.backend is None:
            raise Exception('Please specify backend by set_backend() function before using it.')
        return cls.backend

    @classmethod
    def get_name(cls):
        if cls.backend is None:
            raise Exception('Please specify backend by set_backend() function before using it.')
        return cls.name