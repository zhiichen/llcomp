
import unittest

from Tools.Dump import Dump

from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor
from Backends.Cuda.Mutators.CM_OmpParallel import CM_OmpParallel

from Frontend.Parse import parse_source

from Backends.Cuda.Writers.CUDAWriter import CUDAWriter

from Frontend.InternalRepr import AstToIR


from Backends.Common.tests.common import TestCase

BACKEND_NAME = 'Cuda'
TEST_PATH = 'Backends/' + BACKEND_NAME + '/tests/'
CODE_PATH =  TEST_PATH + 'codes/'
TREE_PATH =  TEST_PATH + 'trees/'

    

class TestCudaFunctions(TestCase):

     def setUp(self):
          pass

     def test_pi(self):
          """ Test basic pi source """
          template_code = open(CODE_PATH + '/pi.c', 'r').read()
          ast = parse_source(template_code, 'pi_test')
          new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
          good_tree = Dump.load(TREE_PATH + '/pi_tree')
          self.check_output(new_ast, good_tree) 

     def test_picu(self):
          """ Test mutating pi to cuda """
          template_code = open(CODE_PATH + '/pi.c', 'r').read()
          ast = parse_source(template_code, 'pi_test')
          tmp = AstToIR(Writer = CUDAWriter).transform(ast)
          new_ast = CM_OmpParallelFor().apply_all(tmp)
          good_tree = Dump.load(TREE_PATH + '/picu_tree')
          self.check_output(new_ast, good_tree)


     def test_mandel(self):
          """ Test mandel source """
          template_code = open(CODE_PATH + '/mandel.c', 'r').read()
          ast = parse_source(template_code, 'mandel_test')
          new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
          good_tree = Dump.load(TREE_PATH + '/mandel_tree')
          self.check_output(new_ast, good_tree)

     def test_mandelcu(self):
          """ Test mutating mandel to cuda """
          template_code = open(CODE_PATH + '/mandel.c', 'r').read()
          ast = parse_source(template_code, 'mandel_test')
          tmp = AstToIR(Writer = CUDAWriter).transform(ast)
          new_ast = CM_OmpParallelFor().apply_all(tmp)
          good_tree = Dump.load(TREE_PATH + '/mandelcu_tree')
          self.check_output(new_ast, good_tree)


     def test_jacobicu(self):
          """ Test mutating mandel to cuda """
          template_code = open(CODE_PATH + '/jacobi.c', 'r').read()
          ast = parse_source(template_code, 'jacobi_test')
          tmp = AstToIR(Writer = CUDAWriter).transform(ast)
          new_ast = CM_OmpParallel().apply_all(tmp)
          good_tree = Dump.load(TREE_PATH + '/jacobicu_tree')
          self.check_output(new_ast, good_tree)


if __name__ == '__main__':
     unittest.main()
