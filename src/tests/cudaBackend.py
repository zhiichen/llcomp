
import unittest

import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


from Tools.Dump import Dump

from Mutators.Cuda import CM_OmpParallelFor

from Tools.Parse import parse_source


from Visitors.clone_visitor import CUDAWriter

from Frontend.InternalRepr import AstToIR



def build_test_trees():
    # Pi
    template_code = open('examples/pi.c', 'r').read()
    ast = parse_source(template_code, 'pi_test')
    new_ast = AstToIR(writer = CUDAWriter).transform(ast)
    Dump.save('tests/pi_tree', ast)
    # CUDA version
    new_ast = CM_OmpParallelFor().apply_all(new_ast)
    Dump.save('tests/picu_tree', new_ast)
    # Mandel
    template_code = open('examples/mandel.c', 'r').read()
    ast = parse_source(template_code, 'mandel_test')
    new_ast = AstToIR(writer = CUDAWriter).transform(ast)
    Dump.save('tests/mandel_tree', new_ast)
    # CUDA version
    new_ast = CM_OmpParallelFor().apply_all(ast)
    Dump.save('tests/mandelcu_tree', new_ast)

   

class TestCudaBackendFunctions(unittest.TestCase):

    def setUp(self):
        self.good_tree = None 

    def test_pi(self):
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
#        link_all_parents(ast)
        new_ast = AstToIR(writer = CUDAWriter).transform(ast)
        self.good_tree = Dump.load('tests/pi_tree')
        ast_str = StringIO();
        good_str = StringIO();
        ast.show(ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(ast_str.getvalue(), good_str.getvalue())

    def test_picu(self):
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_source(template_code, 'pi_test')
        new_ast = AstToIR(writer = CUDAWriter).transform(ast)
#        link_all_parents(ast)
        new_ast = CM_OmpParallelFor().apply_all(new_ast)

        self.good_tree = Dump.load('tests/picu_tree')
        new_ast_str = StringIO();
        good_str = StringIO();
        new_ast.show(new_ast_str)
        self.good_tree.show(good_str)
        print "*** New "
        new_ast.show()
        print "*** Original"
        self.good_tree.show()
        print "*** Finish"
        self.assertEqual(new_ast_str.getvalue(), good_str.getvalue())

    def test_mandel(self):
        template_code = open('examples/mandel.c', 'r').read()
        ast = parse_source(template_code, 'mandel_test')
#        link_all_parents(ast)
        new_ast = AstToIR(writer = CUDAWriter).transform(ast)
        self.good_tree = Dump.load('tests/mandel_tree')
        ast_str = StringIO();
        good_str = StringIO();
        ast.show(ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(ast_str.getvalue(), good_str.getvalue())
 

if __name__ == '__main__':
    build_test_trees()
    unittest.main()
