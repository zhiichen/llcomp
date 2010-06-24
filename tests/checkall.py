""" Run all tests

"""
import unittest
from  Backends.CBackend.tests  import basic
from  Backends.CudaBackend.tests import  cudaBackend, buildCudaTests

import sys
import getopt

# Check if we are building tests... (we only check by default)
COMMAND_LINE_OPTIONS = getopt.getopt(sys.argv[1:], '', ['create'])

if ('--create', '') in COMMAND_LINE_OPTIONS[0]:
    print " Building test trees "
    basic.build_test_trees()
    buildCudaTests.build_pi_tree()
    buildCudaTests.build_mandel_tree()
    buildCudaTests.build_jacobi_tree()

print " Running basic parser tests "
TEST_SUITE = unittest.TestLoader().loadTestsFromTestCase(
   basic.TestParserFunctions)
unittest.TextTestRunner(verbosity=2).run(TEST_SUITE)

print " Running Backend tests "
TEST_SUITE = unittest.TestLoader().loadTestsFromTestCase(
   cudaBackend.TestCudaBackendFunctions)
unittest.TextTestRunner(verbosity=2).run(TEST_SUITE)
