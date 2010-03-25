
import unittest

import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


from Tools.Dump import Dump


def parse_template(template_code, template_name):
        p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P -I /home/rreyes/llcomp/src/include/fake_libc_include/", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        clean_source = p.communicate(template_code)[0]
        process = subprocess.Popen("sed -nf nocomments.sed", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        stripped_code = process.communicate(clean_source)[0]
        ast = c_parser.CParser(lex_optimize = False, yacc_optimize = False).parse(stripped_code, filename = template_name)
	return ast


def build_test_trees():
    template_code = """ 
            int main (int a) {
               printf(" Hello World!");
            }
        """
    ast = parse_template(template_code, 'helloWorld_test')
    Dump.save('tests/helloWorld_tree', ast)
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
    ast = parse_template(template_code, 'pragma_test')
    Dump.save('tests/pragma_tree', ast)



class TestParserFunctions(unittest.TestCase):

    def setUp(self):
        self.good_tree = None 

    def test_helloWorld(self):
        template_code = """ 
            int main (int a) {
               printf(" Hello World!");
            }
        """
	ast = parse_template(template_code, 'helloWorld_test')

        self.good_tree = Dump.load('tests/helloWorld_tree')
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
	ast = parse_template(template_code, 'pragma_test')

        self.good_tree = Dump.load('tests/pragma_tree')
	ast_str = StringIO();
	good_str = StringIO();
	ast.show(ast_str)
	self.good_tree.show(good_str)
        self.assertEqual(ast_str.getvalue(), good_str.getvalue())

	

if __name__ == '__main__':
    build_test_trees()
    unittest.main()
