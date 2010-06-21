
import unittest
from  tests import basic, cudaBackend, buildCudaTests

import sys
import getopt

# Check if we are building tests... (we only check by default)
opts = getopt.getopt(sys.argv[1:], '', ['create'])

if ('--create', '') in opts[0]:
   print " Building test trees "
   basic.build_test_trees()
   buildCudaTests.build_pi_tree()
   buildCudaTests.build_mandel_tree()

print " Running basic parser tests "
suite = unittest.TestLoader().loadTestsFromTestCase(basic.TestParserFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

print " Running Backend tests "
suite = unittest.TestLoader().loadTestsFromTestCase(cudaBackend.TestCudaBackendFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
