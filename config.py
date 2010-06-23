
# Config file for llcomp



# Current work dir
WORKDIR="/home/rreyes/llcomp/"

# Location of cuda files
CUDA_INSTALL_DIR="/usr/local/cuda/"

# C includes for templates
# TODO: Automatically detect include dirs
INCLUDE_DIR = WORKDIR + 'Backends/CudaBackend/Templates/include/'

# Fake stdlib
FAKE_LIBC = WORKDIR + 'Backends/CudaBackend/Templates/include/fake_libc_include'

# Location of pycparser
PYCPARSER_DIR="/home/rreyes/workspace/pycparser-read-only/"

from config_local import *
