
from Tools.Dump import Dump

from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor
from Backends.Cuda.Mutators.CM_OmpParallel import CM_OmpParallel

from Frontend.Parse import parse_source

from Backends.Cuda.Writers.CUDAWriter import CUDAWriter

from Frontend.InternalRepr import AstToIR

BACKEND_NAME = 'Cuda'
TEST_PATH = 'Backends/' + BACKEND_NAME + '/tests/'
CODE_PATH =  TEST_PATH + 'codes/'
TREE_PATH =  TEST_PATH + 'trees/'

def build_mandel_tree():
     """ Builds the tests AST for both mandel.c and mandel.cu 
      
     """
#     [CODE_PATH, TREE_PATH] = getPath(
     # Mandel
     template_code = open(CODE_PATH + '/mandel.c', 'r').read()
     ast = parse_source(template_code, 'mandel_test')
     Dump.save(TREE_PATH + '/mandel_tree', ast)
     tmp = AstToIR(Writer = CUDAWriter).transform(ast)
     # CUDA version
     new_ast = CM_OmpParallelFor().apply_all(tmp) 
     Dump.save(TREE_PATH + '/mandelcu_tree', new_ast)

def build_pi_tree():
     """ Builds the tests AST for both pi.c and pi.cu 
      
     """
     # Pi
     template_code = open(CODE_PATH + '/pi.c', 'r').read()
     ast = parse_source(template_code, 'pi_test')
     Dump.save(TREE_PATH + '/pi_tree', ast)
     # CUDA version
     new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
     tmp = CM_OmpParallelFor().apply_all(new_ast)
     Dump.save(TREE_PATH + '/picu_tree', tmp)

def build_jacobi_tree():
     """ Builds the tests AST for both pi.c and pi.cu 
      
     """
     # Pi
     template_code = open(CODE_PATH + '/jacobi.c', 'r').read()
     ast = parse_source(template_code, 'pi_test')
     Dump.save(TREE_PATH + '/jacobi_tree', ast)
     # CUDA version
     new_ast = AstToIR(Writer = CUDAWriter).transform(ast)
     tmp = CM_OmpParallel().apply_all(new_ast)
     Dump.save(TREE_PATH + '/jacobicu_tree', tmp)

