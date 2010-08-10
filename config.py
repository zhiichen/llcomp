# Config file for llcomp

# Current work dir
WORKDIR="/home/user/llcomp/"
# Location of cuda files
CUDA_INSTALL_DIR="/usr/local/cuda/"
# Location of pycparser
PYCPARSER_DIR="/home/user/pycparser/"


# Import local settings from user

from config_local import *

# C includes for templates
# TODO: Automatically detect include dirs
INCLUDE_DIR = WORKDIR + 'Backends/Cuda/Templates/include/'
# Fake stdlib
FAKE_LIBC = WORKDIR + 'Backends/Cuda/Templates/include/fake_libc_include'




