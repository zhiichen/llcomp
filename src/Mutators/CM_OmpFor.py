from pycparser import c_parser, c_ast
from Visitors.generic_visitors import IDFilter, FuncCallFilter, FuncDeclOfNameFilter, OmpForFilter, OmpParallelFilter
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator

from string import Template

# Copy substructures
import copy


from Mutators.Cuda import CudaMutator


class CM_OmpFor(CudaMutator):
   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the OmpFor node of the AST
      f = OmpForFilter()
      node = f.apply(ast)
      self._func_def = f.get_func_def()
      self._parallel = f.get_parallel()
      return node

   def buildDeclarations(self, numThreads, reduction_node_list):
      """ Builds the declaration section 
          @param numThreads number of threads
          @return Declarations subtree
      """ 
      # Position in the template for dimA declaration, just in case we change it
      DIMA_POS = 0
      MEMSIZE_POS = 4
      if not Dump.exists('Declarations' + self.kernel_name):
         constant_template_code = """
              int dimA = 1;
              int numThreadsPerBlock = 512;
              int numBlocks = dimA / numThreadsPerBlock + (dimA % numThreadsPerBlock?1:0);
              int numElems = numBlocks * numThreadsPerBlock;
              int memSize = numElems * sizeof(double);
        /*    double *reduction_loc_varname;
              double *reduction_cu_varname;  
              */
         """
         tree = self.parse_snippet(constant_template_code, None, name = 'Declarations' + self.kernel_name)
         Dump.save('Declarations' + self.kernel_name, tree)
      else:
         print " Loading frozen template of Declarations " + self.kernel_name
         tree = Dump.load('Declarations' + self.kernel_name)
      declarations =  tree
      reduction_pointer_decls = copy.deepcopy(reduction_node_list)
      # Set Type of reduction_decls  for memSize sizeof (All of the reduction vars must be of the same type)
      declarations.ext[MEMSIZE_POS].init.right.expr.type = reduction_pointer_decls[0].type
      reduction_cu_pointer_decls = self._build_reduction_decls(reduction_pointer_decls)
      # Insert into tree
      declarations.ext.extend(reduction_pointer_decls)
      declarations.ext.extend(reduction_cu_pointer_decls)
      declarations.ext[DIMA_POS].init = numThreads
      return declarations 



   def buildInitialization(self, reduction_vars, ast):
      """ Initialization """
#      _self.build_reduction_malloc_lines()  
            # Template source
      template_code = """
      #include "llcomp_cuda.h"
      int fake() {
      """ + self._build_reduction_malloc_lines(ast, reduction_vars) + "\n}"
   
      return self.parse_snippet(template_code, None, name = 'SendData').ext[-1].body


   def mutatorFunction(self, ast, ompFor_node):
      """ CUDA mutator, writes memory transfer operations for a parallel region
      """
      maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host

      clause_dict = self._get_dict_from_clauses(ompFor_node.clauses,  ast)
      parent_clause_dict = self._get_dict_from_clauses(self._parallel.clauses, ast)
      reduction_params = clause_dict['REDUCTION']
      nowait = clause_dict.has_key('NOWAIT')
      # Private declarations come from the parent parallel construct
      private_params = parent_clause_dict['PRIVATE']
      # shared_params = [ elem for elem in parent_clause_dict['SHARED'] if not isinstance(elem.type, c_ast.ArrayDecl) ]
      shared_params = parent_clause_dict['SHARED']

      ##################### Declarations

      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params)
      InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')
      # Initialization
      initialization_subtree = self.buildInitialization(reduction_vars = reduction_params, ast = ast)

      InsertTool(subtree = initialization_subtree, position = "begin").apply(cuda_stmts, 'stmts')
      

      ##################### Cuda Kernel 

      # Kernel
      kernel_subtree = self.buildKernel(shared_list = shared_params, 
                        private_list = private_params, 
                        reduction_list = reduction_params,
                        loop = ompFor_node.stmt, ast = ast)

      # Function declaration
      # - Build a node without body
      tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
      tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
      kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
      
#      # Find container function
      InsertTool(subtree = kernel_decl, position = "begin", node = self._func_def).apply(ast, 'ext')
      # Function definition
      InsertTool(subtree = kernel_subtree, position = "end" ).apply(ast, 'ext')

      # Support subtree
      # support_subtree = self.buildSupport()
      # InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      # InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
   

      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(reduction_vars = reduction_params, shared_vars = shared_params, ast = ompFor_node)
      InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve(reduction_vars = reduction_params)
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_vars = reduction_params)
      InsertTool(subtree = reduction_subtree, position = "end").apply(cuda_stmts, 'stmts')

      # Replace the entire pragma by a CompoundStatement with all the new statements
#      DotDebugTool(highlight = [ompFor_node]).apply(self._parallel)
      if isinstance(ompFor_node.parent.parent, c_ast.Compound) :
         ReplaceTool(new_node = cuda_stmts, old_node = ompFor_node.parent).apply(ompFor_node.parent.parent, 'stmts')
      else:
         # Maybe we have an error here
         ompFor_node.parent.parent.stmt = cuda_stmts




