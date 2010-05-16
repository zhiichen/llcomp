from pycparser import c_parser, c_ast
from Visitors.generic_visitors import IDFilter, FuncCallFilter, FuncDeclOfNameFilter, OmpForFilter, OmpParallelFilter, FilterError
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Tools.Parse import parse_template
from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator
from Mutators.AbstractMutator import IgnoreMutationException

#from string import Template
from mako.template import Template

import subprocess
from cStringIO import StringIO

# Copy substructures
import copy

class CudaMutatorError(Exception):
   def __init__(self, description):
      self.description = description

   def __str__(self):
      return "CudaMutatorError :: " + self.description


class CudaMutator(object):
   """ This  mutator locates a omp parallel for reduction, and then
      translate the original source to an equivalent cuda implementation 
   """
   def __init__(self, clauses = {}, kernel_name = 'loopKernel'):
      """ Constructor """
      # BUG: Don't work with optimize
      self.template_parser = c_parser.CParser(lex_optimize = False, yacc_optimize = False)
      self.kernel_name = kernel_name
      self._func_def = None
      self._parallel = None
      self._clauses = clauses

   # TODO: Clean this function, it has an strange behaviour
   def get_names(self, elem, ast):
      """ Return a list of names for a type """
      type = type_of_id(elem, ast)
      while not hasattr(type, 'names') and not hasattr(type, 'name'):
         # TODO: Launch exception if not type attribute
         type = type.type
      if hasattr(type, 'names'):
         return type.names
      else:
         return [type.name]

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the OmpFor node of the AST
      f = OmpForFilter()
      node = f.apply(ast)
      self._func_def = f.get_func_def()
      self._parallel = f.get_parallel()
      return node

   def getThreadNum(self, node):
      """ Gets the maximum number of threads needed """
      if node.op == '<' or node.op == '<=':
          return node.right
      else:
          return node.left

   def parse_snippet(self, template_code, subs_dir, name):
      subtree = None
 #     template_code = Template(template_code).substitute(subs_dir)
      if subs_dir:
         template_code = Template(template_code).render(**subs_dir)
 #        print " Tempalte " + str(template_code)
      try:
         subtree = parse_template(template_code, name)
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)

	 return None
      except IOError:
         print "Pipe Error"
	 return None
      return subtree;

   def _get_dict_from_clauses(self, clauses, ast):
      """ Return a dict of clauses from a list of OmpClause objects
         
           Example: [OmpClause('REDUCTION', ...), OmpClause('PRIVATE', ...)]
             will return:  {'REDUCTION' : [....] , 'PRIVATE' : [...]}
      """
      clause_names = ['SHARED', 'PRIVATE', 'NOWAIT', 'REDUCTION']
      clause_dict = self._clauses
      # Note: Each identifiers is a ParamList
      for elem in clauses:
         if not clause_dict.has_key(elem.name):
               clause_dict[elem.name] = []
         
         if elem.name == 'SHARED':
            for id in elem.identifiers.params:
               clause_dict[elem.name].append(decl_of_id(id, ast))
         elif elem.name == 'PRIVATE':
            for id in elem.identifiers.params:
               clause_dict[elem.name].append(decl_of_id(id, ast))
         elif elem.name == 'NOWAIT':
            clause_dict[elem.name] = True
         elif elem.name == 'REDUCTION':
            for id in elem.identifiers.params:
               clause_dict[elem.name].append(decl_of_id(id, ast))

      for name in clause_names:
         if not clause_dict.has_key(name):
            clause_dict[name] = []

      self._clauses = clause_dict
      return  clause_dict

   def _build_shared_memory_decls_cu(self, shared_node_list, parent, ast = None):
   # Build shared memory declarations on host
      tmp = copy.deepcopy(shared_node_list)
      shared_cu_pointer_decls = []
      for elem in tmp:
         if isinstance(elem.type, c_ast.ArrayDecl):
            ptr = c_ast.Decl(elem.name + '_cu', elem.quals, [], elem.type.type, None, None, None, parent)
            PointerMutator().apply(ptr)
            shared_cu_pointer_decls.append(ptr)
         elif isinstance(elem.type, c_ast.Struct):
            IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name +'_cu')).apply_all(elem)
            PointerMutator().apply(elem)
      return shared_cu_pointer_decls


   def _build_shared_memory_init_cu(self, shared_node_list, ast):
      shared_dict = {} 
      for elem in shared_node_list:
         # Only malloc / send if it is a complex type
         if isinstance(elem.type, c_ast.ArrayDecl): 
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value
         elif isinstance(elem.type, c_ast.Struct):
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ")"

      shared_malloc_lines = "\n".join(["cudaMalloc((void **) &" + str(key) + "_cu," + str(value) + ");" for key,value in shared_dict.items()])
      shared_malloc_lines += "\n".join(["cudaMemcpy(" + str(key) + "_cu," + str(key) + ", " + str(value) + ", cudaMemcpyHostToDevice);" for key,value in shared_dict.items()])

      return shared_malloc_lines


   def _build_reduction_decls(self, reduction_pointer_decls):
      reduction_cu_pointer_decls = copy.deepcopy(reduction_pointer_decls)
      # Build local reduction vars
      for elem in reduction_pointer_decls:
         IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_loc_' + elem.name)).apply_all(elem)
         PointerMutator().apply(elem)
      # Build cuda reduction arrays
      for elem in reduction_cu_pointer_decls:
         IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_cu_' + elem.name)).apply_all(elem)
         PointerMutator().apply(elem)
      return reduction_cu_pointer_decls


   def _build_reduction_malloc_lines(self, ast, reduction_vars):
      reduction_dict = {} 
      # Host memory allocation (malloc lines)
      for elem in reduction_vars:
          reduction_dict[str(elem.name)] = self.get_names(elem, ast)[0]
      reduction_malloc_lines = "\n".join(["reduction_loc_" + str(key) + " = (" + str(value) +"*) malloc(numElems * sizeof(" + str(value) + "));" for key,value in reduction_dict.items()])
      reduction_dict = {} 


      # Device memory allocation (cudaMalloc lines)
      for elem in reduction_vars:
         reduction_dict[str(elem.name)] = "sizeof(" + "reduction_cu_".join(self.get_names(elem, ast)) +")"
      reduction_malloc_lines += "\n".join(["cudaMalloc((void **) &" + "reduction_cu_" + str(key) + ", numElems * " + str(value) + ");" for key,value in reduction_dict.items()])
      # TODO: Initial value for * reduction must be 1 instead of 0
      reduction_malloc_lines += "\n".join(["cudaMemset(reduction_cu_" + str(key) + ", (int)" + str(key) + ", numElems * " + str(value) + ");" for key,value in reduction_dict.items()])
      return reduction_malloc_lines



   def buildDeclarations(self, numThreads, reduction_node_list, shared_node_list):
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
      # Maybe we haven't got a reduction clause
      if len(reduction_node_list):
         reduction_pointer_decls = copy.deepcopy(reduction_node_list)
         # Set Type of reduction_decls  for memSize sizeof (All of the reduction vars must be of the same type)
         declarations.ext[MEMSIZE_POS].init.right.expr.type = reduction_pointer_decls[0].type
         reduction_cu_pointer_decls = self._build_reduction_decls(reduction_pointer_decls)
         # Insert into tree
         declarations.ext.extend(reduction_pointer_decls)
         declarations.ext.extend(reduction_cu_pointer_decls)
      declarations.ext.extend(self._build_shared_memory_decls_cu(shared_node_list, declarations))
      declarations.ext[DIMA_POS].init = numThreads
      return declarations 


   def buildInitialization(self, reduction_vars, shared_vars, ast):
      """ Initialization """
      reduction_dict = {} 
 
      shared_dict = {} 
      for elem in shared_vars:
         # Only malloc / send if it is a complex type
         if isinstance(elem.type, c_ast.ArrayDecl): 
            # print "Array Decl: " + elem.name
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value
         elif isinstance(elem.type, c_ast.Struct):
            shared_dict[elem.name] = "sizeof(" + " ".join(self.get_names(elem, ast)) +  ")"

      shared_malloc_lines = "\n".join(["cudaMalloc((void **) &" + str(key) + "_cu," + str(value) + ");" for key,value in shared_dict.items()])
      shared_malloc_lines += "\n".join(["cudaMemcpy(" + str(key) + "_cu," + str(key) + ", " + str(value) + ", cudaMemcpyHostToDevice);" for key,value in shared_dict.items()])
      # Template source
      template_code = """
      #include "llcomp_cuda.h" 
      int fake() {
      """ + shared_malloc_lines  + self._build_reduction_malloc_lines(ast, reduction_vars) + "\n}"
   
      return self.parse_snippet(template_code, None, name = 'SendData').ext[-1].body

   def buildRetrieve(self, reduction_vars, modified_shared_vars, ast = None):
      memcpy_lines = []
      # If a shared var is modified inside kernel, we retrieve it from device
      for elem in modified_shared_vars:
         # Only malloc / send if it is a complex type
#         if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
#            memcpy_lines += [elem.name]
         if isinstance(elem.type, c_ast.ArrayDecl): 
            memcpy_lines.append([ elem.name, "sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value])
         elif isinstance(elem.type, c_ast.Struct):
            memcpy_lines.append([elem.name,  "sizeof(" + " ".join(self.get_names(elem, ast)) +  ")"])

      # Template source
      template_code = """
      int fake() {
      % for name in reduction_names:
        cudaMemcpy(reduction_loc_${name}, reduction_cu_${name}, memSize, cudaMemcpyDeviceToHost);
      % endfor

      % for elem in shared_names:
        cudaMemcpy(${elem[0]}, ${elem[0]}_cu, ${elem[1]}, cudaMemcpyDeviceToHost);
      % endfor

      checkCUDAError("memcpy");
      }
      """ 
      return self.parse_snippet(template_code, {'reduction_names' : reduction_vars, 'shared_names' : memcpy_lines}, name = 'Retrieve').ext[0].body
      
   def buildKernelLaunch(self, reduction_vars, shared_vars, ast):
       # FIXME : reduction_vars is now an array of declarations
       reduction_var_list = ",".join("reduction_cu_" + elem.name for elem in reduction_vars)
       # shared_var_list = ",".join(elem.name + "_cu" for elem in shared_vars)
       shared_var_list = [];
       for elem in shared_vars:
         # Only malloc / send if it is a complex type
         elem_type = type_of_id(elem, ast)
         ptr =""
         # Check if it is a pointer
         # if isinstance(type, c_ast.PtrDecl):
         #   elem_type = elem_type.type
         #   ptr = "* "
         if isinstance(elem_type, c_ast.ArrayDecl) or isinstance(elem_type, c_ast.Struct): 
            shared_var_list += [ptr + str(elem.name) + "_cu"]
         else:
            shared_var_list += [ptr + str(elem.name)]


       kernel_parameters = " "
       if len(reduction_var_list) > 0:
         kernel_parameters += reduction_var_list
       if len(reduction_var_list) > 0 and len(shared_var_list) > 0:
         kernel_parameters += ","
       if len(shared_var_list) > 0:
         kernel_parameters += ",".join(shared_var_list)

       template_code = """
  	#include "llcomp_cuda.h" 

       int fake() {
              dim3 dimGrid (numBlocks);
     	        dim3 dimBlock (numThreadsPerBlock);

          ${kernelName} <<< dimGrid, dimBlock >>> (${kernelParameters});
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, {'kernelParameters' : kernel_parameters,  'kernelName' : self.kernel_name}, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
       return tree


   def buildHostReduction(self, reduction_vars, ast):
      if len(reduction_vars) == 0:
         return c_ast.Compound(stmts = [], decls = [])

      reduction_lines = ""
      free_lines = ""
      for elem in reduction_vars:
         reduction_lines += elem.name + "+= reduction_loc_" + (elem.name) + "[__i__];\n"
         free_lines += "cudaFree(reduction_cu_" + (elem.name) + ");\n"
         free_lines += "free(reduction_loc_" + (elem.name) + ");\n"
      # TODO: Add shared vars to free

      wait_lines = "cudaThreadSynchronize();\n";

      template_code = """
      int fake() {
/*      #define LLC_REDUCTION_FUNC(dest, fuente) dest = dest + fuente*/
      ${var} = kernelReduction_${type}(reduction_cu_${var}, numElems, ${var});

      /* By default, omp for has a wait at the end */
      ${wait}
      ${free_lines}
      }
      """
      return self.parse_snippet(template_code, {'var' : elem.name, 'type' : self.get_names(elem, ast)[0], 'free_lines' : free_lines, 'wait' : wait_lines}, name = 'HostReduction').ext[0].body

   def buildSupport(self):
      """ CUDA Support subroutines """
      if not Dump.exists('SupportRoutines'):
         template_code = """
        #include "llcomp_cuda.h"  

void checkCUDAError (const char *msg)
{
	cudaError_t err = cudaGetLastError ();
	if (cudaSuccess != err)
	{
		fprintf (stderr, "Cuda error: %s: %s.\\n", msg,
				cudaGetErrorString (err));
		exit (EXIT_FAILURE);
	}
}


"""
         print " Parsing template of Support Routines "
         tree = self.parse_snippet(template_code, None, name = 'SupportRoutines')
         Dump.save('SupportRoutines', tree)
      else:
         print " Loading frozen template of SupportRoutines "
         tree = Dump.load('SupportRoutines')

      return c_ast.Compound(stmts = [tree.ext[-1]], decls = [tree.ext[-1].decl])

   def buildKernel(self, shared_list, private_list, reduction_list, loop, ast):
      if not Dump.exists(self.kernel_name):
          template_code = """
         #include "llcomp_cuda.h" 
          __global__ void ${kernelName} (double * reduction_cu)
          {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          ;
          }
          """
          print " Parsing template of KernelBuild "
          tree = self.parse_snippet(template_code, {'kernelName' : self.kernel_name}, name = 'KernelBuild')
          Dump.save(self.kernel_name, tree)
      else:
          print " Loading frozen template of KernelBuild "
          tree = Dump.load(self.kernel_name)

      # Note: we need to mutate the declaration subtree into a param declaration (ArrayRef to Pointer and so on...)
      # Check if we have reductions
      if reduction_list:
         reduction_vars = copy.deepcopy(reduction_list)
         for elem in reduction_vars:
            # Replace the name of the declaration. The type_declaration doesn't change, so maybe we'll get problems later?
            IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_cu_' + elem.name)).apply_all(elem)
            PointerMutator().apply(elem)
         # Add the declarations to the parameters of the functions
         DeclsToParamsMutator(decls = reduction_vars).apply(tree.ext[-1].function.decl.type.args)
      
      # OpenMP shared vars are parameters of the kernel function
      if shared_list:
         shared_vars = copy.deepcopy(shared_list)
         # TODO: Move this to DeclsToParamsMutator
         for elem in shared_vars:
            # Replace the name of the declaration. 
            if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
               mut = IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name + '_cu'))
               mut.apply_all(elem)
               mut.apply_all(loop.stmt)
               # Pointer instead of array 
               elem.type = elem.type.type
               PointerMutator().apply(elem)
         # Add the declarations to the parameters of the functions
         DeclsToParamsMutator(decls = shared_vars).apply(tree.ext[-1].function.decl.type.args)

      # Remove the template declaration
      RemoveTool(tree.ext[-1].function.decl.type.args.params[0]).apply(tree.ext[-1].function.decl.type.args, 'params')

      # OpenMP Private vars need to be declared inside kernel
      #    - we build a tmp Compound to group all declarations, and insert them once
      tmp = c_ast.Compound(decls= private_list, stmts=[])
      #    - Insert tool removes the parent node of the inserted subtree
      InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')

      # Add the loop statements, (but not the reduction)
      IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx')).apply_all(loop.stmt)
      # Add the loop statements, (but not the reduction)
      IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx')).apply_all(loop.cond)

      # Identify function calls inside kernel and replace the definitions to __device__ 
      try:
         for func_call in FuncCallFilter().iterate(loop.stmt):
           # func_call.show()
           # DotDebugTool(highlight = [func_call]).apply(loop.stmt)
           # print " Writing " + func_call.name.name + " to device "
           try:
              fcm = FuncToDeviceMutator(func_call = func_call).apply(ast)
           except IgnoreMutationException as ime:
              # This function is already implemented on device, so we continue we don't need to convert it
              print "CudaMutator:: Warning :: " + str(ime)
      except NodeNotFound:
         # There are not function calls on the loop.stmt
         pass
      except FilterError as fe:
         raise CudaMutatorError(fe.get_description())
      # Identify function calls inside kernel and replace the definitions to __device__ 
      # TODO: This is incorrect, we should write a subtree instead of a bare string...
      for elem in reduction_list:
         IDNameMutator(old = c_ast.ID(name = elem.name, parent = elem.parent), new = c_ast.ID(name = 'reduction_cu_' + str(elem.name) + '[idx]', parent = elem.parent)).apply_all(loop.stmt)
      # Insert the code inside kernel
      # We need to check if the idx is inside for limits (in case we have more threads than iterations)
      check_boundary_node = c_ast.Compound(decls = None, stmts = [c_ast.If(cond = loop.cond, iftrue = loop.stmt, iffalse = None)], parent = tree.ext[-1].function.body)
      InsertTool(subtree = check_boundary_node, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
      return c_ast.FileAST(ext = [tree.ext[-1]])

   def mutatorFunction(self, ast, ompFor_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
#      filter = FilterVisitor(match_node_type = c_astFor, prev_brother = ompFor_node)
      # parallelFor = ompFor_node.stmt
      # Parent of the node
      # parent_stmt = self._parallel
      # container_func = self._func_def
      # Maximum number of parallel threads
#      from Tools.Debug import DotDebugTool

      maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)

#      DotDebugTool(select_node = maxThreadNumber_node).apply(parallelFor.cond)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host

      clause_dict = self._get_dict_from_clauses(ompFor_node.clauses,  ast)
      shared_params = clause_dict['SHARED']
      
      private_params = clause_dict['PRIVATE']
      reduction_params = clause_dict['REDUCTION']
      nowait = clause_dict.has_key('NOWAIT')

      ##################### Declarations

      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params, shared_node_list = shared_params)
      InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')
      # Initialization
      initialization_subtree = self.buildInitialization(reduction_vars = reduction_params, shared_vars = shared_params, ast = ast)

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
      support_subtree = self.buildSupport()
      InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
   

      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(reduction_vars = reduction_params, shared_vars = shared_params, ast = ompFor_node)
      InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Retrieve data
      # TODO : Detect modified vars
      retrieve_subtree = self.buildRetrieve(reduction_vars = reduction_params, modified_shared_vars = shared_params)
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_vars = reduction_params, ast = ast)
      InsertTool(subtree = reduction_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Replace the entire pragma by a CompoundStatement with all the new statements
      ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')


      ##################### Final tree operations

      # Remove the pragma from the destination code
#      RemoveTool(target_node = self._parallel).apply(container_func, 'stmts')



   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try: 
         start_node = self.filter(ast)
         self.mutatorFunction(ast, start_node)
         # Remove pragma from code
      except NodeNotFound as nf:
         print nf
      return ast




