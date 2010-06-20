
import unittest

import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


from Tools.Dump import Dump

from Mutators.Cuda import CM_OmpParallelFor

from Tools.Parse import parse_source


from Visitors.clone_visitor import CUDAWriter

from Frontend.InternalRepr import AstToIR


class TestCase(unittest.TestCase):
   def __init__(self, *args, **kwargs):
      super(TestCase, self).__init__(*args, **kwargs)

   def check_output(self, new_tree, good_tree):
      import difflib
      import pprint
      # Build differ object and calculate diferences
      d = difflib.Differ()
      diff = list(d.compare(str(good_tree).split('\n'), str(new_tree).split('\n')))
      # If differences exists, show them
      if len([elem for elem in diff if elem[0] != ' ']) != 0:
         pprint.pprint(diff)
      self.assertEqual(str(good_tree).split('\n'), str(new_tree).split('\n'))


def build_mandel_tree():
    # Mandel
    template_code = open('examples/mandel.c', 'r').read()
    ast = parse_source(template_code, 'mandel_test')
    Dump.save('tests/mandel_tree', ast)
    tmp = AstToIR(writer = CUDAWriter).transform(ast)
    # CUDA version
    new_ast = CM_OmpParallelFor().apply_all(tmp)
    Dump.save('tests/mandelcu_tree', new_ast)



def build_test_trees():
    # Pi
    template_code = open('examples/pi.c', 'r').read()
    ast = parse_source(template_code, 'pi_test')
    Dump.save('tests/pi_tree', ast)
    # CUDA version
    new_ast = AstToIR(writer = CUDAWriter).transform(ast)
    tmp = CM_OmpParallelFor().apply_all(new_ast)
    Dump.save('tests/picu_tree', tmp)
#    build_mandel_tree()


   

class TestCudaBackendFunctions(TestCase):

    def setUp(self):
        self.good_tree = None 

    def test_pi(self):
        """ Test basic pi source """
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
        new_ast = AstToIR(writer = CUDAWriter).transform(ast)
        good_tree = Dump.load('tests/pi_tree')
        self.check_output(new_ast, good_tree) 

    def test_picu(self):
        """ Test mutating pi to cuda """
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
        tmp = AstToIR(writer = CUDAWriter).transform(ast)
        new_ast = CM_OmpParallelFor().apply_all(tmp)
        good_tree = Dump.load('tests/picu_tree')
        self.check_output(new_ast, good_tree)


#    def test_mandel(self):
#        """ Test mandel source """
#        template_code = open('examples/mandel.c', 'r').read()
#        ast = parse_source(template_code, 'mandel_test')
#        new_ast = AstToIR(writer = CUDAWriter).transform(ast)
#        good_tree = Dump.load('tests/mandel_tree')
#        self.check_output(new_ast, good_tree)

if __name__ == '__main__':
    build_test_trees()
    unittest.main()
