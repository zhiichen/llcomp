
import unittest

import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


from Frontend.Parse import parse_source
from Tools.Dump import Dump



def build_test_trees():
     template_code = """ 
                int main (int a) {
                    printf(" Hello World!");
                }
          """
     ast = parse_source(template_code, 'helloWorld_test')
     Dump.save('Backends/CBackend/tests/trees/helloWorld_tree', ast)
     template_code = """ 
int main()
{
     int i;
     int sum[10];

     for (i = 0; i <= 10; i++) {
    sum[i] = i;
     }

     #pragma omp parallel for reduction(+ : sum)
     for (i = 0; i <= 10; i++) {
    sum[i] = i;
     }

}
          """
     ast = parse_source(template_code, 'pragma_test')
     Dump.save('Backends/CBackend/tests/trees/pragma_tree', ast)

     template_code = open('Backends/CBackend/tests/codes/jacobi_big.c', 'r').read()
     ast = parse_source(template_code, 'jacobi_c')
     Dump.save('Backends/CBackend/tests/trees/jacobi_tree', ast)




class TestParserFunctions(unittest.TestCase):

     def setUp(self):
          self.good_tree = None 

     def test_helloWorld(self):
          template_code = """ 
                int main (int a) {
                    printf(" Hello World!");
                }
          """
          ast = parse_source(template_code, 'helloWorld_test')

          self.good_tree = Dump.load('Backends/CBackend/tests/trees/helloWorld_tree')
          ast_str = StringIO();
          good_str = StringIO();
          ast.show(ast_str)
          self.good_tree.show(good_str)
          self.assertEqual(ast_str.getvalue(), good_str.getvalue())

     def test_pragma(self):
          template_code = """
            int main()
            {
            int i;
            int sum[10];
            for (i = 0; i <= 10; i++) {
                sum[i] = i;
            }
            #pragma omp parallel for reduction(+ : sum)
            for (i = 0; i <= 10; i++) {
                sum[i] = i;
            }
            }
            """
          ast = parse_source(template_code, 'pragma_test')
          self.good_tree = Dump.load('Backends/CBackend/tests/trees/pragma_tree')
          ast_str = StringIO();
          good_str = StringIO();
          ast.show(ast_str)
          self.good_tree.show(good_str)
          self.assertEqual(ast_str.getvalue(), good_str.getvalue())

     def test_jacobi(self):
         template_code = open('Backends/CBackend/tests/codes/jacobi_big.c', 'r').read()
         ast = parse_source(template_code, 'jacobi_test')
         self.good_tree = Dump.load('Backends/CBackend/tests/trees/jacobi_tree')
         ast_str = StringIO();
         good_str = StringIO();
         ast.show(ast_str)
         self.good_tree.show(good_str)
         self.assertEqual(ast_str.getvalue(), good_str.getvalue())
 

if __name__ == '__main__':
     build_test_trees()
     unittest.main()
