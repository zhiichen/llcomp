
from pycparser import c_ast


from Backends.Common.Visitors.GenericVisitors import *

from Backends.CudaBackend.Visitors.CM_Visitors import OmpForFilter, OmpParallelFilter, OmpThreadPrivateFilter

from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool

from Tools.Declarations import decl_of_id

from Tools.Debug import DotDebugTool

from Backends.Common.Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator

from Backends.CudaBackend.Mutators.Common import AbstractCudaMutator

from Backends.Common.TemplateEngine.TemplateParser import TemplateParser, get_template_array

class CM_OmpParallel(AbstractCudaMutator):
    def filter(self, ast):
        """ Filter definition
            Returns the first node matching with the filter"""
        # Build a visitor , matching the OmpFor node of the AST
        f = OmpParallelFilter(device = self.device)
        node = f.apply(ast)
        self._func_def = f.get_func_def()
        self._parallel = node
        return node

    def apply_all(self, ast):
        """ Apply mutation to all matches """
        start_node = None
        self.ast = ast
        f = OmpParallelFilter(device = self.device)
        num = 0;
        self.kernel_name = self.kernel_prefix
        old_prefix = self.kernel_prefix
        old_parallel = self._parallel
        try:
            for elem in f.iterate(ast):
                # Save previous state
                self.kernel_name = self.kernel_name + str(num)
                self.kernel_prefix = self.kernel_prefix + str(num)
                self._clauses = {}
                # Current scope variables
                self._func_def = f.get_func_def()
                self._target_device_node = f._target_device_node
                self._clauses = self._get_dict_from_clauses(f._target_device_node.clauses, ast)
                self._parallel = elem
                start_node = self.mutatorFunction(ast, elem)
                # Restore previous state
                # TODO : Maybe a parent relink is needed?
                self._parallel = old_parallel
                self.kernel_prefix = old_prefix
                self.kernel_name = old_prefix
                num += 1;
        except NodeNotFound as nf:
            print str(nf)
        except StopIteration:
            return self._parallel
        return ast



    def buildParallelDeclarations(self, shared_node_list, ast):
        """ Builds the declaration section of a Parallel Region
                This code handles memory transfers between host and cuda
              
                @param shared_node_list List of shared variable declarations
                @param ast Full ast (for type search)

             @return Parallel Declarations subtree
        """ 
        # Position in the template for dimA declaration, just in case we change it
        tmp = []
        for elem in shared_node_list: 
            if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type,c_ast.Struct):
                tmp.append(elem)

        shared_vars = get_template_array(tmp, ast) 

        # Type string | var name | pointer to type | pointer to var | declaration string
        template_code = """
            int main() {
                 % for var in shared_vars:
                        ${var.type} * ${var.name}_cu;
                  % endfor
                  % for var in shared_vars:
                  cudaMalloc((void **) (&${var.name}_cu), ${var.numelems} * sizeof(${var.type}));
                  cudaMemcpy(${var.name}_cu, ${var.name}, ${var.numelems} * sizeof(${var.type}), cudaMemcpyHostToDevice); 
                  % endfor
            }

            """
        print "New kernel build with name : " + self.kernel_name
        parallel_init = self.parse_snippet(template_code, {'shared_vars' : shared_vars}, name = 'Initialization of Parallel Region ' + self.kernel_name, show = False).ext[-1].body
#~     from Tools.Debug import DotDebugTool
#~     DotDebugTool().apply(kernel_init)
        return parallel_init


    def mutatorFunction(self, ast, ompParallel_node):
        """ CUDA mutator, writes memory transfer operations for a parallel region
        """
        from Backends.CudaBackend.Mutators.CM_OmpFor import CM_OmpFor

        threadprivate = []
        for elem in OmpThreadPrivateFilter().dfs_iter(ast):
            threadprivate.extend([decl_of_id(it, ast) for it in elem.identifiers.params])
        # print " Threadprivate : " + str(threadprivate)

        clause_dict = self._get_dict_from_clauses(ompParallel_node.clauses, ast)

        shared_params = clause_dict['SHARED']
        modified_shared_vars = clause_dict['COPY_OUT']
        copyin_shared_vars = clause_dict['COPY_IN']

        private_params = clause_dict['PRIVATE'] 
        nowait = clause_dict.has_key('NOWAIT')
        # If the parallel statement have declarations, they are private to the thread, so, we need to put them as params
        if ompParallel_node.stmt.decls:
            private_params += ompParallel_node.stmt.decls
        private_params.extend(threadprivate)

        # Loops inside parallel region (wired for now)
        CM_OmpFor(clause_dict, kernel_name = self.kernel_prefix + "_loopKernel").apply_all(ompParallel_node, ast)

        ##################### Statement for cuda
        cuda_stmts = c_ast.Compound(stmts = [], decls = []);

        ##################### Cuda parameters on host


        ##################### Declarations

        declarations_subtree = self.buildParallelDeclarations(shared_node_list = copyin_shared_vars, ast = ast)
        InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')

        # Initialization
#        initialization_subtree = self.buildInitialization(shared_vars = shared_params, ast = ast)
        InsertTool(subtree = self._parallel.stmt, position = "begin").apply(cuda_stmts, 'stmts')
#        InsertTool(subtree = initialization_subtree, position = "begin").apply(cuda_stmts, 'stmts')

        # Retrieve data
        retrieve_subtree = self.buildRetrieve(reduction_vars = [], modified_shared_vars = modified_shared_vars, ast = ast, shared_vars = copyin_shared_vars)
        InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')


        ##################### Support subtree
        ### Support has been moved to an external include file
        # support_subtree = self.buildSupport()
        # InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
        # InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')
        print " *** You must include reduction_snippets.h after translation *** "


        ##################### Loop substitution 
    
        # Replace the entire pragma by a CompoundStatement with all the new statements
        # Note: The parent of Parallel is always a Pragma node
        ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')


