
import unittest

import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


from Tools.Dump import Dump

from Mutators.Cuda import CudaMutator

def parse_template(template_code, template_name):
        p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P -I /home/rreyes/llcomp/src/include/fake_libc_include/", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        clean_source = p.communicate(template_code)[0]
        process = subprocess.Popen("sed -nf nocomments.sed", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stripped_code = process.communicate(clean_source)[0]
        ast = c_parser.CParser(lex_optimize = False, yacc_optimize = False).parse(stripped_code, filename = template_name)
        return ast

def link_all_parents(ast):
        """ Function to link the nodes of the AST in reverse order, using a parent attribute in each node """
        def deep_first_search(root, visited = None,
                preorder_process  = lambda x: None):
                """
                Given a starting vertex, root, do a depth-first search.
                """
                to_visit = [] 
                if visited is None: visited = set()

                to_visit.append(root) # Start with root
                while len(to_visit) != 0:
                        v = to_visit.pop()
                        if v not in visited:
                                visited.add(v)
                                preorder_process(v)
                                to_visit.extend(v.children())
        def link_parent(node):
                for child in node.children():
                        child.parent = node

        deep_first_search(root = ast, visited = None, preorder_process = link_parent)




def build_test_trees():
    # Pi
    template_code = open('examples/pi.c', 'r').read()
    ast = parse_template(template_code, 'pi_test')
    link_all_parents(ast)
    Dump.save('tests/pi_tree', ast)
    # CUDA version
    new_ast = CudaMutator().apply(ast)
    Dump.save('tests/picu_tree', new_ast)
    # Mandel
    template_code = open('examples/mandel.c', 'r').read()
    ast = parse_template(template_code, 'mandel_test')
    link_all_parents(ast)
    Dump.save('tests/mandel_tree', ast)
    # CUDA version
    new_ast = CudaMutator().apply(ast)
    Dump.save('tests/mandelcu_tree', new_ast)

   

class TestCudaMutatorFunctions(unittest.TestCase):

    def setUp(self):
        self.good_tree = None 

    def test_pi(self):
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_template(template_code, 'pi_test')
        link_all_parents(ast)
        self.good_tree = Dump.load('tests/pi_tree')
        ast_str = StringIO();
        good_str = StringIO();
        ast.show(ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(ast_str.getvalue(), good_str.getvalue())

    def test_picu(self):
        template_code = open('examples/pi.c', 'r').read()
        ast = parse_template(template_code, 'pi_test')
        link_all_parents(ast)
        new_ast = CudaMutator().apply(ast)

        self.good_tree = Dump.load('tests/picu_tree')
        new_ast_str = StringIO();
        good_str = StringIO();
        new_ast.show(new_ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(new_ast_str.getvalue(), good_str.getvalue())

    def test_mandel(self):
        template_code = open('examples/mandel.c', 'r').read()
        ast = parse_template(template_code, 'mandel_test')
        link_all_parents(ast)
        self.good_tree = Dump.load('tests/mandel_tree')
        ast_str = StringIO();
        good_str = StringIO();
        ast.show(ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(ast_str.getvalue(), good_str.getvalue())
 
    def test_mandelcu(self):
        template_code = open('examples/mandel.c', 'r').read()
        ast = parse_template(template_code, 'mandel_test')
        link_all_parents(ast)
        new_ast = CudaMutator().apply(ast)

        self.good_tree = Dump.load('tests/mandelcu_tree')
        new_ast_str = StringIO();
        good_str = StringIO();
        new_ast.show(new_ast_str)
        self.good_tree.show(good_str)
        self.assertEqual(new_ast_str.getvalue(), good_str.getvalue())
     

if __name__ == '__main__':
    build_test_trees()
    unittest.main()
