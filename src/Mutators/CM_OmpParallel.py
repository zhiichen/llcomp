
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
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value
         elif isinstance(elem.type, c_ast.Struct):
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ")"

      shared_malloc_lines = "\n".join(["cudaMalloc((void **) &" + str(key) + "_cu," + str(value) + ");" for key,value in shared_dict.items()])
      shared_malloc_lines += "\n".join(["cudaMemcpy(" + str(key) + "_cu," + str(key) + ", " + str(value) + ", cudaMemcpyHostToDevice);" for key,value in shared_dict.items()])
      # Template source
      template_code = """
      #include "llcomp_cuda.h"
      int fake() {
      """ + shared_malloc_lines + "\n}"
   
      return self.parse_snippet(template_code, None, name = 'SendData').ext[-1].body

   def buildRetrieve(self, modified_shared_vars, ast):
      memcpy_lines = ""
      # CudaMemCpy lines 
      for elem in modified_shared_vars:
 #        memcpy_lines += "cudaMemcpy(reduction_loc_" + (elem.name) + ", reduction_cu_" + elem.name + ", memSize, cudaMemcpyDeviceToHost);\n"
         # Only malloc / recieve if it is a complex type
         if isinstance(elem.type, c_ast.ArrayDecl): 
            memcpy_lines += "cudaMemcpy(" + str(elem.name) + ", " + str(elem.name) + "_cu, sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value + ", cudaMemcpyDeviceToHost);"
         elif isinstance(elem.type, c_ast.Struct):
            memcpy_lines += "cudaMemcpy(" + str(elem.name) + ", " + str(elem.name) + "_cu, sizeof(" + " ".join(self.get_names(elem, ast)) +  "), cudaMemcpyDeviceToHost);"

     
      # Template source
      template_code = """
      int fake() {
/*      cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost); */
        $cudaMemcpyLines
      checkCUDAError("memcpy");
      }
      """ 

      return self.parse_snippet(template_code, {'cudaMemcpyLines' : memcpy_lines}, name = 'Retrieve').ext[0].body


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

      # Loops inside parallel region (wired for now)
 #     CM_OmpFor(clause_dict, kernel_name = 'initKernel').apply(ast)
  #    CM_OmpFor(clause_dict, kernel_name = 'loopKernel').apply(ast)
      CM_OmpFor(clause_dict, kernel_name = 'loopKernel').apply_all(ompParallel_node, ast)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host


      ##################### Declarations

      declarations_subtree = self.buildDeclarations(shared_node_list = shared_params, ast = ast)
      InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')

      # Initialization
      initialization_subtree = self.buildInitialization(shared_vars = shared_params, ast = ast)
      InsertTool(subtree = self._parallel.stmt, position = "begin").apply(cuda_stmts, 'stmts')
      InsertTool(subtree = initialization_subtree, position = "begin").apply(cuda_stmts, 'stmts')

      # Retrieve data
      retrieve_subtree = self.buildRetrieve(modified_shared_vars = shared_params, ast = ast)
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')


      ##################### Support subtree
      support_subtree = self.buildSupport()
      InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
   
      # Replace the entire pragma by a CompoundStatement with all the new statements
      # Note: The parent of Parallel is always a Pragma node
      ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')


