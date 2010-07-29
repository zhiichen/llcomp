from pycparser import c_parser, c_ast


from Backends.Cuda.Visitors.CM_Visitors import *

from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.Declarations import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Backends.Common.Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator


from Backends.Cuda.Mutators.Common import AbstractCudaMutator

from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor


from Backends.Common.TemplateEngine.TemplateParser import TemplateParser, get_template_array

class CM_OmpFor(CM_OmpParallelFor):

    def filter(self, ast):
        """ Filter definition
            Returns the first node matching with the filter"""
        # Build a visitor , matching the OmpFor node of the AST
        f = OmpForFilter()
        node = f.apply(ast)
        self._func_def = f.get_func_def()
        self._parallel = f.get_parallel()
        return node

    def apply_all(self, parent_parallel_node, ast):
        """ Apply mutation to all matches """
        start_node = None
        self.ast = ast
        f = OmpForFilter()
        num = 0;
 #          start_node = self.filter(ast)
 #          self.mutatorFunction(ast, start_node)

        try:
            for elem in f.iterate(ast):
                # Save previous state
                old_name = self.kernel_name
                old_clauses = self._clauses
                self.kernel_name = self.kernel_name + str(num)
                # Current scope variables
                self._func_def = f.get_func_def()
                self._parallel = f.get_parallel()
                # If current node is not child of first parallel node, stop
                if self._parallel != parent_parallel_node:
                    raise StopIteration
                start_node = self.mutatorFunction(ast, elem)
                # Restore previous state
                self.kernel_name = old_name
                self._clauses = old_clauses
                num += 1;
        except NodeNotFound as nf:
            print str(nf)
        except StopIteration:
            return self._parallel
        return start_node

    def buildRetrieve(self, reduction_vars, modified_shared_vars):
        memcpy_lines = ""
        # CudaMemCpy lines 
        #for elem in reduction_vars:
        #    memcpy_lines += "cudaMemcpy(reduction_loc_" + (elem.name) + ", reduction_cu_" + elem.name + ", memSize, cudaMemcpyDeviceToHost);\n"
      
        # Template source
        template_code = """
        int fake() {
/*        cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost); */
          ${cudaMemcpyLines}
        checkCUDAError("memcpy");
        }
        """ 

        return self.parse_snippet(template_code, {'cudaMemcpyLines' : memcpy_lines}, name = 'Retrieve').ext[0].body


    def mutatorFunction(self, ast, ompFor_node):
        """ CUDA mutator, writes memory transfer operations for a parallel region
        """
        maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)

        ##################### Statement for cuda
        cuda_stmts = c_ast.Compound(stmts = [], decls = []);

        ##################### Cuda parameters on host

        clause_dict = self._get_dict_from_clauses(ompFor_node.clauses,  ast)
  
        reduction_params = clause_dict['REDUCTION']
        nowait = clause_dict.has_key('NOWAIT')
        # Private declarations come from the parent parallel construct
        private_params = clause_dict['PRIVATE']
        shared_params = clause_dict['SHARED']


        ##################### Declarations

        kernel_init_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params, shared_node_list = [], ast = ast)
        InsertTool(subtree = kernel_init_subtree, position = "begin").apply(cuda_stmts, 'decls')
        

        ##################### Cuda Kernel 

        # Kernel
        kernel_subtree = self.buildKernel(shared_list = shared_params, 
                                private_list = private_params, 
                                reduction_list = reduction_params,
                                loop = ompFor_node.stmt, ast = ast)

        # Function declaration
        # - Build a node without body
#        tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
#        tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
#        kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
        
#        # Find container function
#        InsertTool(subtree = kernel_decl, position = "begin", node = self._func_def).apply(ast, 'ext')
        # Function definition
        InsertTool(subtree = kernel_subtree, position = "begin", node = self._func_def ).apply(ast, 'ext')

        # Support subtree
        # support_subtree = self.buildSupport()
        # InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
        # InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


        ##################### Loop substitution 
    

        # Kernel Launch
        kernelLaunch_subtree = self.buildKernelLaunch(reduction_vars = reduction_params, shared_vars = shared_params, ast = ompFor_node)
        InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
        # Retrieve data
        retrieve_subtree = self.buildRetrieve(reduction_vars = reduction_params, modified_shared_vars = shared_params)
        InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
        # Host reduction
        reduction_subtree = self.buildHostReduction(reduction_vars = reduction_params, ast = ast)
        InsertTool(subtree = reduction_subtree, position = "end").apply(cuda_stmts, 'stmts')

        # Replace the entire pragma by a CompoundStatement with all the new statements
#        DotDebugTool(highlight = [ompFor_node]).apply(self._parallel)
        if isinstance(ompFor_node.parent.parent, c_ast.Compound) :
            ReplaceTool(new_node = cuda_stmts, old_node = ompFor_node.parent).apply(ompFor_node.parent.parent, 'stmts')
        else:
            # Maybe we have an error here
            ompFor_node.parent.parent.stmt = cuda_stmts




