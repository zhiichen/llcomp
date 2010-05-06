
from pycparser import c_parser, c_ast
from Visitors.generic_visitors import IDFilter, FuncCallFilter, FuncDeclOfNameFilter, OmpForFilter, OmpParallelFilter, OmpThreadPrivateFilter
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator

from string import Template



from Mutators.Cuda import CudaMutator

class CM_OmpParallel(CudaMutator):
   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the OmpFor node of the AST
      f = OmpParallelFilter()
      node = f.apply(ast)
      self._func_def = f.get_func_def()
      self._parallel = node
      return node

   def buildDeclarations(self, shared_node_list, ast):
      """ Builds the declaration section 
          @return Declarations subtree
      """ 
      # Position in the template for dimA declaration, just in case we change it
      declarations =  c_ast.FileAST(ext = [])
      declarations.ext.extend(self._build_shared_memory_decls_cu(shared_node_list, declarations, ast))
      return declarations 

   def buildInitialization(self, shared_vars, ast):
      """ Initialization """
      reduction_dict = {} 
 
      shared_dict = {} 
      for elem in shared_vars:
         # Only malloc / send if it is a complex type
         if isinstance(elem.type, c_ast.ArrayDecl): 
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(ast,elem)) +  ") * " +  elem.type.dim.value
         elif isinstance(elem.type, c_ast.Struct):
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(ast,elem)) +  ")"

      shared_malloc_lines = "\n".join(["cudaMalloc((void **) &" + str(key) + "_cu," + str(value) + ");" for key,value in shared_dict.items()])
      shared_malloc_lines += "\n".join(["cudaMemcpy(" + str(key) + "_cu," + str(key) + ", " + str(value) + ", cudaMemcpyHostToDevice);" for key,value in shared_dict.items()])
      # Template source
      template_code = """
      #include "llcomp_cuda.h"
      int fake() {
      """ + shared_malloc_lines + "\n}"
   
      return self.parse_snippet(template_code, None, name = 'SendData').ext[-1].body


   def mutatorFunction(self, ast, ompParallel_node):
      """ CUDA mutator, writes memory transfer operations for a parallel region
      """
      from Mutators.CM_OmpFor import CM_OmpFor

      threadprivate = []
      for elem in OmpThreadPrivateFilter().dfs_iter(ast):
         threadprivate.extend([decl_of_id(it, ast) for it in elem.identifiers.params])
      print " Threadprivate : " + str(threadprivate)

      clause_dict = self._get_dict_from_clauses(ompParallel_node.clauses, ast)
      shared_params = clause_dict['SHARED']
      private_params = clause_dict['PRIVATE'] 
      nowait = clause_dict.has_key('NOWAIT')
      # If the parallel statement have declarations, they are private to the thread, so, we need to put them as params
      if ompParallel_node.stmt.decls:
         private_params += ompParallel_node.stmt.decls
      private_params.extend(threadprivate)


      CM_OmpFor(clause_dict).apply(ast)

#      from Tools.Debug import DotDebugTool
#      DotDebugTool().apply(ast.ext[18])


      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host
#      print " *** Private params :  " 
#      for priv in private_params:
#         priv.show()

#      DotDebugTool().apply(private_params)

      ##################### Declarations

      declarations_subtree = self.buildDeclarations(shared_node_list = shared_params, ast = ast)
      InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')

      # Initialization
      initialization_subtree = self.buildInitialization(shared_vars = shared_params, ast = ast)
      InsertTool(subtree = self._parallel.stmt, position = "begin").apply(cuda_stmts, 'stmts')
      InsertTool(subtree = initialization_subtree, position = "begin").apply(cuda_stmts, 'stmts')

      ##################### Support subtree
      support_subtree = self.buildSupport()
      InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
   
      # Replace the entire pragma by a CompoundStatement with all the new statements
      # Note: The parent of Parallel is always a Pragma node
      ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')
      # DotDebugTool().apply(self._parallel)
      # DotDebugTool().apply(cuda_stmts)
      # InsertTool(subtree = cuda_stmts, position = "begin").apply(self._parallel.stmt, 'stmts')


