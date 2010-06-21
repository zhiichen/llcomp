
from Tools.Dump import Dump

from Mutators.Cuda import CM_OmpParallelFor

from Tools.Parse import parse_source

from Visitors.clone_visitor import CUDAWriter

from Frontend.InternalRepr import AstToIR


def build_mandel_tree():
    """ Builds the tests AST for both mandel.c and mandel.cu 
     
         Stores the tests ast under FREEZER_DIR/tests/mandel{,cu}_tree
    """

    # Mandel
    template_code = open('examples/mandel.c', 'r').read()
    ast = parse_source(template_code, 'mandel_test')
    Dump.save('tests/mandel_tree', ast)
    tmp = AstToIR(Writer = CUDAWriter).transform(ast)
    # CUDA version
    new_ast = CM_OmpParallelFor().apply_all(tmp) # .apply_all(tmp)
    Dump.save('tests/mandelcu_tree', new_ast)

def build_pi_tree():
    """ Builds the tests AST for both pi.c and pi.cu 
     
         Stores the tests ast under FREEZER_DIR/tests/pi{,cu}_tree
    """
    # Pi
    template_code = open('examples/pi.c', 'r').read()
    ast = parse_source(template_code, 'pi_test')
    Dump.save('tests/pi_tree', ast)
    # CUDA version
    new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
    tmp = CM_OmpParallelFor().apply_all(new_ast)
    Dump.save('tests/picu_tree', tmp)


