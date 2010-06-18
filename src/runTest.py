
import unittest
from  tests import basic, cudaBackend


# Uncomment to create new tests
# basic.build_test_trees()
cudaBackend.build_test_trees()


print " Running basic parser tests "
suite = unittest.TestLoader().loadTestsFromTestCase(basic.TestParserFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

print " Running Backend tests "
suite = unittest.TestLoader().loadTestsFromTestCase(cudaBackend.TestCudaBackendFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
