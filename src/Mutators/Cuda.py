from pycparser import c_parser, c_ast
from Visitors.generic_visitors import FilterVisitor, IDFilter, FuncCallFilter, FuncDeclOfNameFilter
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator

from string import Template

import subprocess
from cStringIO import StringIO

# Copy substructures
import copy


class CudaMutator(object):
   """ This  mutator locates a omp parallel for reduction, and then
      translate the original source to an equivalent cuda implementation 
   """
   def __init__(self):
      " Constructor "
      # BUG: Don't work with optimize
      self.template_parser = c_parser.CParser(lex_optimize = False, yacc_optimize = False)
      self.kernel_name = 'reductionKernel'

   def get_names(ast,self, elem):
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
      # Build a visitor , matching the Pragma node of the AST
      f = FilterVisitor(match_node_type = c_ast.Pragma)
      node = f.apply(ast)
      return node

   def getThreadNum(self, node):
      """ Gets the maximum number of threads needed """
      if node.op == '<' or node.op == '<=':
          return node.right
      else:
          return node.left

   def parse_snippet(self, template_code, subs_dir, name):
      subtree = None
      template_code = Template(template_code).substitute(subs_dir)
      try:
         p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P -I /home/rreyes/llcomp/src/include/ 2>/dev/null", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
         clean_source = p.communicate(template_code)[0]
         process = subprocess.Popen("sed -nf nocomments.sed", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
         stripped_code = process.communicate(clean_source)[0]
         subtree = self.template_parser.parse(stripped_code, filename=name)
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)

	 return None
      except IOError:
         print "Pipe Error"
	 return None
      return subtree;


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
      reduction_pointer_decls = copy.deepcopy(reduction_node_list)
      # Set Type of reduction_decls  for memSize sizeof (All of the reduction vars must be of the same type)
      declarations.ext[MEMSIZE_POS].init.right.expr.type = reduction_pointer_decls[0].type
      # Build local reduction vars
      for elem in reduction_pointer_decls:
         IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_loc_' + elem.name)).apply_all(elem)
         PointerMutator().apply(elem)
      # Build cuda reduction arrays
      reduction_cu_pointer_decls = copy.deepcopy(reduction_node_list)
      for elem in reduction_cu_pointer_decls:
         IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_cu_' + elem.name)).apply_all(elem)
         PointerMutator().apply(elem)
      # Build shared memory declarations on host
      tmp = copy.deepcopy(shared_node_list)
      shared_cu_pointer_decls = []
      for elem in tmp:
         if isinstance(elem.type, c_ast.ArrayDecl):
            ptr = c_ast.Decl(elem.name + '_cu', elem.quals, [], elem.type.type, None, None, None, declarations)
            PointerMutator().apply(ptr)
            shared_cu_pointer_decls.append(ptr)
         elif isinstance(elem.type, c_ast.Struct):
            IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name +'_cu')).apply_all(elem)
            PointerMutator().apply(elem)

      # Insert into tree
      declarations.ext.extend(reduction_pointer_decls)
      declarations.ext.extend(reduction_cu_pointer_decls)
      declarations.ext.extend(shared_cu_pointer_decls)
      declarations.ext[DIMA_POS].init = numThreads
      return declarations 


   def buildInitializaton(self, reduction_vars, shared_vars, ast):
      """ Initialization """
      reduction_dict = {} 
      # Host memory allocation (malloc lines)
      for elem in reduction_vars:
          reduction_dict[str(elem.name)] = self.get_names(ast,elem)[0]
      reduction_malloc_lines = "\n".join(["reduction_loc_" + str(key) + " = (" + str(value) +"*) malloc(numElems * sizeof(" + str(value) + "));" for key,value in reduction_dict.items()])
      reduction_dict = {} 


      # Device memory allocation (cudaMalloc lines)
      for elem in reduction_vars:
         reduction_dict[str(elem.name)] = "sizeof(" + "reduction_cu_".join(self.get_names(ast,elem)) +")"
      reduction_malloc_lines += "\n".join(["cudaMalloc((void **) &" + "reduction_cu_" + str(key) + ", numElems * " + str(value) + ");" for key,value in reduction_dict.items()])
      # TODO: Initial value for * reduction must be 1 instead of 0
      # TODO: Maybe initial value should be the init value of the reduction var?
#      reduction_malloc_lines += "\n".join(["cudaMemset((void **) &reduction_cu_" + str(key) + ", (int)" + str(key) + ", numElems * " + str(value) + ");" for key,value in reduction_dict.items()])
      reduction_malloc_lines += "\n".join(["cudaMemset(reduction_cu_" + str(key) + ", (int)" + str(key) + ", numElems * " + str(value) + ");" for key,value in reduction_dict.items()])

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
      """ + shared_malloc_lines  + reduction_malloc_lines + "\n}"
   
      return self.parse_snippet(template_code, None, name = 'SendData').ext[-1].body

   def buildRetrieve(self, reduction_vars):
      memcpy_lines = ""
      # CudaMemCpy lines 
      for elem in reduction_vars:
         memcpy_lines += "cudaMemcpy(reduction_loc_" + (elem.name) + ", reduction_cu_" + elem.name + ", memSize, cudaMemcpyDeviceToHost);\n"
      
      # Template source
      template_code = """
      int fake() {
/*      cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost); */
        $cudaMemcpyLines
      checkCUDAError("memcpy");
      }
      """ 

      return self.parse_snippet(template_code, {'cudaMemcpyLines' : memcpy_lines}, name = 'Retrieve').ext[0].body
      
   def buildKernelLaunch(self, reduction_vars, shared_vars, ast):
       reduction_var_list = ",".join("reduction_cu_" + elem.name for elem in reduction_vars)
       # shared_var_list = ",".join(elem.name + "_cu" for elem in shared_vars)
       shared_var_list = [];
       for elem in shared_vars:
         # Only malloc / send if it is a complex type
         if isinstance(type_of_id(elem, ast), c_ast.ArrayDecl) or isinstance(type_of_id(elem, ast), c_ast.Struct): 
            shared_var_list += [str(elem.name) + "_cu"]
         else:
            shared_var_list += [str(elem.name)]

       template_code = """
  	#include "llcomp_cuda.h"

       int fake() {
              dim3 dimGrid (numBlocks);
     	        dim3 dimBlock (numThreadsPerBlock);


   	    $kernelName <<< dimGrid , dimBlock >>> ($reductionvars, $sharedvars);
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, {'reductionvars' : reduction_var_list, 'sharedvars' : ",".join(shared_var_list), 'kernelName' : self.kernel_name}, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
       return tree


   def buildHostReduction(self, reduction_vars):
      reduction_lines = ""
      free_lines = ""
      for elem in reduction_vars:
         reduction_lines += elem.name + "+= reduction_loc_" + (elem.name) + "[i];\n"
         free_lines += "cudaFree(reduction_cu_" + (elem.name) + ");\n"
         free_lines += "free(reduction_loc_" + (elem.name) + ");\n"
      # TODO: Add shared vars to free

      wait_lines = "cudaThreadSynchronize();\n";

      template_code = """
      int fake() {
      for (i = 0; i < dimA; i++)
      {
          $reduction_lines
      }
      /* By default, omp for has a wait at the end */
      $wait
      $free_lines
      }
      """
      return self.parse_snippet(template_code, {'reduction_lines' : reduction_lines, 'free_lines' : free_lines, 'wait' : wait_lines}, name = 'HostReduction').ext[0].body

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
          __global__ void $kernelName (double * reduction_cu)
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
      # OpenMP shared vars are parameters of the kernel function
      # Note: we need to mutate the declaration subtree into a param declaration (ArrayRef to Pointer and so on...)
      reduction_vars = copy.deepcopy(reduction_list)
      for elem in reduction_vars:
         # Replace the name of the declaration. The type_declaration doesn't change, so maybe we'll get problems later?
         IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID('reduction_cu_' + elem.name)).apply_all(elem)
         PointerMutator().apply(elem)
      # Add the declarations to the parameters of the functions
      DeclsToParamsMutator(decls = reduction_vars).apply(tree.ext[-1].function.decl.type.args)
      DeclsToParamsMutator(decls = shared_list).apply(tree.ext[-1].function.decl.type.args)
      # Remove the template declaration
      RemoveTool(tree.ext[-1].function.decl.type.args.params[0]).apply(tree.ext[-1].function.decl.type.args, 'params')
      # OpenMP Private vars need to be declared inside kernel
      #    - we build a tmp Compound to group all declarations, and insert them once
      tmp = c_ast.Compound(decls= private_list, stmts=None)
      #    - Insert tool removes the parent node of the inserted subtree
      InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')
      # Add the loop statements, (but not the reduction)

#~      from Tools.Debug import DotDebugTool
#~      DotDebugTool().apply(loop.stmt)
      IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx')).apply_all(loop.stmt)
#~      DotDebugTool().apply(loop.stmt)
      # Identify function calls inside kernel and replace the definitions to __device__ 
      try:
         for func_call in FuncCallFilter().iterate(loop.stmt):
           fcm = FuncToDeviceMutator(func_call = func_call).apply(ast)
      except NodeNotFound:
         # There are not function calls on the loop.stmt
         pass
      # Identify function calls inside kernel and replace the definitions to __device__ 
      # TODO: This is incorrect, we should write a subtree instead of a bare string...
      for elem in reduction_list:
         IDNameMutator(old = c_ast.ID(name = elem.name, parent = elem.parent), new = c_ast.ID(name = 'reduction_cu_' + str(elem.name) + '[idx]', parent = elem.parent)).apply_all(loop.stmt)
      InsertTool(subtree = loop.stmt, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
#~      DotDebugTool().apply(loop.stmt)
      return c_ast.FileAST(ext = [tree.ext[-1]])

   def mutatorFunction(self, ast, prev_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
      filter = FilterVisitor(match_node_type = c_ast.For, prev_brother = prev_node)
      parallelFor = filter.apply(ast)
      # Parent of the node
      parent_stmt = filter.parentOfMatch()
      # Maximum number of parallel threads
#      from Tools.Debug import DotDebugTool

      maxThreadNumber_node = self.getThreadNum(parallelFor.cond)

#      DotDebugTool(select_node = maxThreadNumber_node).apply(parallelFor.cond)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host
      
      reduction_vars = prev_node.child.reduction[0].identifiers[0].params
      shared_vars =  prev_node.child.shared[0].identifiers[0].params
      private_vars = prev_node.child.private[0].identifiers[0].params 

      shared_params = [ decl_of_id(elem, ast) for elem in shared_vars ]
      private_params = [ decl_of_id(elem, ast) for elem in private_vars ]
      reduction_params = [ decl_of_id(elem, ast) for elem in reduction_vars ]

      ##################### Declarations

#~      from Tools.Debug import DotDebugTool
#~      DotDebugTool().apply(maxThreadNumber_node)
      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params, shared_node_list = shared_params)
      InsertTool(subtree = declarations_subtree, position = "begin").apply(cuda_stmts, 'decls')
      # Initialization
#      initialization_subtree = self.buildInitializaton(reduction_vars = reduction_params, shared_vars = prev_node.child.shared[0].identifiers[0].params, ast = ast)
      initialization_subtree = self.buildInitializaton(reduction_vars = reduction_params, shared_vars = shared_params, ast = ast)

      InsertTool(subtree = initialization_subtree, position = "begin").apply(cuda_stmts, 'stmts')
      

      ##################### Cuda Kernel 

      # Kernel
      kernel_subtree = self.buildKernel(shared_list = shared_params, 
                        private_list = private_params, 
                        reduction_list = reduction_params,
                        loop = parallelFor, ast = ast)
#      from Tools.Debug import DotDebugTool 
#      DotDebugTool(select_node = ast).apply(kernel_subtree.ext[0])

      # Function declaration
      # - Build a node without body
      tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
      tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
      kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
      InsertTool(subtree = kernel_decl, position = "begin", node = parent_stmt.parent ).apply(ast, 'ext')
      # Function definition
      InsertTool(subtree = kernel_subtree, position = "end" ).apply(ast, 'ext')

      # Support subtree
      support_subtree = self.buildSupport()
      InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
   

      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(reduction_vars = reduction_vars, shared_vars = shared_vars, ast = parent_stmt)
      InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve(reduction_vars = reduction_vars)
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_vars = prev_node.child.reduction[0].identifiers[0].params)
      InsertTool(subtree = reduction_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Replace for by a CompoundStatement with all the new statements
      ReplaceTool(new_node = cuda_stmts, old_node = parallelFor).apply(parent_stmt, 'stmts')


      ##################### Final tree operations

      # Remove the pragma from the destination code
      RemoveTool(target_node = prev_node).apply(parent_stmt, 'stmts')



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
