
import unittest

from Backends.Common.Tools.Dump import Dump

from Backends.CudaBackend.Mutators.CM_OmpParallelFor import CM_OmpParallelFor

from Frontend.Parse import parse_source

from Backends.CudaBackend.Writers.CUDAWriter import CUDAWriter

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


   

class TestCudaBackendFunctions(TestCase):

    def setUp(self):
        pass

    def test_pi(self):
        """ Test basic pi source """
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
        new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
        good_tree = Dump.load('tests/pi_tree')
        self.check_output(new_ast, good_tree) 

    def test_picu(self):
        """ Test mutating pi to cuda """
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
        tmp = AstToIR(Writer = CUDAWriter).transform(ast)
        new_ast = CM_OmpParallelFor().apply_all(tmp)
        good_tree = Dump.load('tests/picu_tree')
        self.check_output(new_ast, good_tree)


    def test_mandel(self):
        """ Test mandel source """
        template_code = open('examples/mandel.c', 'r').read()
        ast = parse_source(template_code, 'mandel_test')
        new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
        good_tree = Dump.load('tests/mandel_tree')
        self.check_output(new_ast, good_tree)

    def test_mandelcu(self):
        """ Test mutating mandel to cuda """
        template_code = open('examples/mandel.c', 'r').read()
        ast = parse_source(template_code, 'mandel_test')
        tmp = AstToIR(Writer = CUDAWriter).transform(ast)
        new_ast = CM_OmpParallelFor().apply_all(tmp)
        good_tree = Dump.load('tests/mandelcu_tree')
        self.check_output(new_ast, good_tree)


if __name__ == '__main__':
    unittest.main()
