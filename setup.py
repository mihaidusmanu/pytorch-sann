from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension('torch_sann.spatially_aware_nn_cuda',
                      ['cuda/spatially_aware_nn.cpp', 'cuda/spatially_aware_nn_kernel.cu']),
 
    ]

__version__ = '0.0.1'
url = 'https://github.com/mihaidusmanu/pytorch-sann'

setup_requires = ['pytest-runner']
install_requires = []
tests_require = ['pytest', 'numpy']

setup(
    name='torch_sann',
    version=__version__,
    description=('Spatially Aware Nearest Neighbors for PyTorch'),
    author='Mihai Dusmanu',
    author_email='mihai.dusmanu@inf.ethz.ch',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'nn', 'features', 'negative-mining', 'local-features', 'guided-matching'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
