
import unittest

from Tools.Dump import Dump

from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor
from Backends.Cuda.Mutators.CM_OmpParallel import CM_OmpParallel

from Frontend.Parse import parse_source

from Backends.C.Writers.OmpWriter import OmpWriter

from Frontend.InternalRepr import AstToIR


BACKEND_NAME = 'Common'
TEST_PATH = 'Backends/' + BACKEND_NAME + '/tests/'
CODE_PATH =  TEST_PATH + 'codes/'
TREE_PATH =  TEST_PATH + 'trees/'

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


    

class TestCommonFunctions(TestCase):

     def setUp(self):
          pass

     def test_Tools(self):
          """ Test common tools """
          template_code = '''
              int main() {
                  int i = 0;
           /*       printf("Hello World!\n");*/
              }
          '''
          ast = parse_source(template_code, "Tool test 1")
          # Transform the C ast into the internal representation
          new_ast = AstToIR(Writer = OmpWriter).transform(ast)
          template_code = """
              int main() {
                  int f = 0;
                  f = 7;
              }
          """
          declarations = AstToIR(Writer = OmpWriter).transform(parse_source(template_code, "Tool test 2")).ext[-1].body.decls
          from Tools.Tree import InsertTool, ReplaceTool, RemoveTool
          InsertTool(subtree = declarations, position = "begin").apply(new_ast.ext[-1].body, 'decls')

          self.assertEqual(new_ast.ext[-1].body.decls[-1].parent, new_ast.ext[-1].body)

          good_tree = Dump.load(TREE_PATH + '/insert_tool_tree')
          self.check_output(new_ast, good_tree) 


if __name__ == '__main__':
     unittest.main()
