
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

def build_tools_tree():
    """ Builds the tests AST for both pi.c and pi.cu 
    
    """
    template_code = '''
    int main() {
        int f = 0;
        int i = 0;
        i = 7;
    }
    '''
    new_ast = parse_source(template_code, "Tool test 1")
#    # Transform the C ast into the internal representation
#    new_ast = AstToIR(Writer = OmpWriter).transform(ast)
#    template_code = """
#        int main() {
#            int f = 0;
#            f = 7;
#        }
#    """
#    declarations = AstToIR(Writer = OmpWriter).transform(parse_source(template_code, "Tool test 2")).ext[-1].body.decls
#    from Tools.Tree import InsertTool, ReplaceTool, RemoveTool
#    InsertTool(subtree = declarations, position = "begin").apply(new_ast.ext[-1].body, 'decls')

    Dump.save(TREE_PATH + '/insert_tool_tree', new_ast)

