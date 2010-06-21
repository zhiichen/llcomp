from pycparser import c_parser, c_ast
from Visitors.generic_visitors import IDFilter, FuncCallFilter, FuncDeclOfNameFilter, OmpForFilter, OmpParallelFilter,  OmpParallelForFilter, FilterError, TypedefFilter, IdentifierTypeFilter
from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool

from Tools.Debug import DotDebugTool

from Frontend.Parse import parse_source

from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator

from TemplateEngine.TemplateParser import TemplateParser, get_template_array


from Visitors.clone_visitor import CWriter


from Mutators.Cuda.Common import AbstractCudaMutator

class CM_OmpParallelFor(AbstractCudaMutator):
   """ This  mutator locates a omp parallel for reduction, and then
      translate the original source to an equivalent cuda implementation 

   """
   def __init__(self, clauses = None, kernel_name = 'loopKernel', kernel_prefix = ''):
      """ Constructor """
      self._ompFor = None
      self._clauses = {}
      super(CM_OmpParallelFor, self).__init__(clauses, kernel_name, kernel_prefix)


   def filter(self, ast):
      """ Filter definition
         
      @return first node matching with the filter
      """
      # Build a visitor , matching the OmpFor node of the AST
      f = OmpParallelForFilter()
      node = f.apply(ast)
      self._func_def = f.get_func_def()
      self._parallel = f.get_parallel()
      return node

   def apply_all(self, ast):
      """ Apply mutation to all matches 
         
         @return last node changed
      """
      start_node = None
      self.ast = ast
      f = OmpParallelForFilter()
      num = 0;
      try:
         for elem in f.iterate(ast):
            # Save previous state
            old_name = self.kernel_name
            old_clauses = self._clauses
            self.kernel_name = self.kernel_name + str(num)
            # Current scope variables
            self._func_def = f.get_func_def()
            self._parallel = f.get_parallel()
            self._ompFor = f.get_ompFor()
            # Apply mutation
            start_node = self.mutatorFunction(ast, self._ompFor)
            # Restore previous state
            self.kernel_name = old_name
            self._clauses = old_clauses
            num += 1;
      except NodeNotFound as nf:
         print str(nf)
      except StopIteration:
         return self._parallel

      return start_node




   def mutatorFunction(self, ast, ompFor_node):
      """ Main mutator for OpenMP Parallel For construct

         Writes the optimized code of an OpenMP Parallel For construct, building a kernel
            overwriting the for loop.
      """
      ##################### Get thread number node
      maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host

      clause_dict = self._get_dict_from_clauses(self._parallel.clauses + ompFor_node.clauses,  ast)
      shared_params = clause_dict['SHARED']
      private_params = clause_dict['PRIVATE']
      reduction_params = clause_dict['REDUCTION']
      nowait = clause_dict.has_key('NOWAIT')

      ##################### Declarations

      kernel_init_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params, shared_node_list = shared_params, ast = ast)
      InsertTool(subtree = kernel_init_subtree, position = "begin").apply(cuda_stmts, 'decls')
           

      ##################### Cuda Kernel 

      # Kernel
      kernel_subtree = self.buildKernel(shared_list = shared_params, 
                        private_list = private_params, 
                        reduction_list = reduction_params,
                        loop = ompFor_node.stmt, ast = ast)

      # Function definition
      InsertTool(subtree = kernel_subtree, position = "begin" ).apply(ast, 'ext')

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
      ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')

      ##################### Final tree operations
      return ast



