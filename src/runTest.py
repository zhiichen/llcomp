
import unittest
from  tests import basic


# basic.build_test_trees()
suite = unittest.TestLoader().loadTestsFromTestCase(basic.TestParserFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
