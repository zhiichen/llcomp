
import unittest
from  tests import basic, cudaMutator


# basic.build_test_trees()


print " Running basic parser tests "
suite = unittest.TestLoader().loadTestsFromTestCase(basic.TestParserFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

print " Running Mutator tests "
suite = unittest.TestLoader().loadTestsFromTestCase(cudaMutator.TestCudaMutatorFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
