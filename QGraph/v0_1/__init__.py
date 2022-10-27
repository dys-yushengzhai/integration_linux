import importlib
import os.path as osp

import torch

__version__ = '0.1'

if True:
    libs = ['libqgraph']
    paths = ['/home/dys/integration/QGraph/v0_1/']
else:
    libs = [
            '_version',  '_graclus'
    ]
    paths = [osp.dirname(__file__)]
for library in libs:
    cuda_spec = importlib.machinery.PathFinder().find_spec(f'{library}_cuda', paths)
    cpu_spec = importlib.machinery.PathFinder().find_spec(f'{library}', paths)
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                        f"{osp.dirname(__file__)}")

cuda_version = torch.ops.torch_cluster.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_cluster were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_cluster has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_cluster that '
            f'matches your PyTorch install.')

from .graph_coarsen import graph_coarsen  

__all__ = [
    'graph_coarsen'
    '__version__',
]
