from pycparser import c_parser, c_ast
from Visitors.generic_visitors import FilterVisitor, IDFilter
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Mutators.DeclsToParams import DeclsToParamsMutator

from string import Template

import subprocess
from cStringIO import StringIO


class IDNameMutator(object):
   """  Replace and ID name with another ID name """
   def __init__(self, old, new):
      self.old = old
      self.new = new
 
   def filter(self, ast):
      id_node = None
      try:
         # ast.show()
         af = IDFilter(id = self.old)
         id_node = af.apply(ast)
      except NodeNotFound:
         # print " *** here *** "
         return None
      return id_node

   def mutatorFunction(self, ast):
      delattr(ast, 'name')
      setattr(ast, 'name', self.new.name)
      return ast

   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try:
         start_node = self.filter(ast)
         if start_node:
            self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node


class CudaMutator(object):
   """ This  mutator locates a omp parallel for reduction, and then
      translate the original source to an equivalent cuda implementation 
   """
   def __init__(self):
      " Constructor "
      # BUG: Don't work with optimize
      self.template_parser = c_parser.CParser(lex_optimize = False, yacc_optimize = False)
      self.kernel_name = 'reductionKernel'

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


   def buildDeclarations(self, numThreads):
      """ Builds the declaration section 
          @param numThreads number of threads
          @return Declarations subtree
      """ 
      template_code = """
      int dimA = $numThreads;
      int numThreadsPerBlock = 512;
      int numBlocks = dimA / numThreadsPerBlock;
      int memSize = numBlocks * numThreadsPerBlock * sizeof (double);
      double *reduction_loc = (double *) malloc (memSize);
      double *reduction_cu;
      """
      return self.parse_snippet(template_code, dict(numThreads = numThreads), name = 'declarations')

   def buildInitializaton(self, shared_vars, ast):
      """ Initialization """
      shared_dict = {} 
      for elem in shared_vars:
          # Only malloc / send if it is a complex type
          if isinstance(elem, c_ast.ArrayDecl) or isinstance(elem, c_ast.Struct):
	          shared_dict[elem.name] = "sizeof(" + " ".join(type_of_id(elem, ast).type.names) +  ")"
      shared_malloc_lines = "\n".join(["cudaMalloc((void **) &" + str(key) + "," + str(value) + ");" for key,value in shared_dict.items()])
      template_code = """
      int fake() {
          cudaMalloc((void **) &reduction_cu, memSize);
      """ + shared_malloc_lines + "\n}"
      return self.parse_snippet(template_code, None, name = 'SendData').ext[0].body

   def buildRetrieve(self):
      template_code = """
      int fake() {
      cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost);
      checkCUDAError("memcpy");
      }
      """ 
      return self.parse_snippet(template_code, None, name = 'Retrieve').ext[0].body
      
   def buildKernelLaunch(self, shared_var_list):
       template_code = """
	#include "llcomp_cuda.h"

       int fake() {
   	    dim3 dimGrid (numBlocks);
   	    dim3 dimBlock (numThreadsPerBlock);
 	       $kernelName <<< dimGrid , dimBlock >>> (reduction_cu, $sharedvars);
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, {'sharedvars' : shared_var_list, 'kernelName' : self.kernel_name}, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
       return tree


   def buildHostReduction(self, reduction_var_node):
      template_code = """
      int fake() {
      for (i = 0; i < dimA; i++)
      {
        $target += reduction_loc[i];
      }
      }
      """
      return self.parse_snippet(template_code, {'target' : reduction_var_node.name}, name = 'HostReduction').ext[0].body

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

   def buildKernel(self, params, private_list, reduction_list, loop, ast):
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
      param_decls = [ decl_of_id(elem, ast) for elem in params ]
      pm = DeclsToParamsMutator(decls = param_decls)
      pm.apply(tree.ext[-1].function.decl.type.args)
      # OpenMP Private vars need to be declared inside kernel
      #    - we build a tmp Compound to group all declarations, and insert them once
      tmp = c_ast.Compound(decls= [decl_of_id(elem, ast) for elem in private_list], stmts=None)
      #    - Insert tool removes the parent node of the inserted subtree
      InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')
      # Add the loop statements, (but not the reduction)
      km = IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx'))
      km.apply(loop.stmt)
      # TODO: This is incorrect, we should write a subtree instead of a bare string...
      km = IDNameMutator(old = c_ast.ID('sum'), new = c_ast.ID('reduction_cu[idx]'))
      km.apply(loop.stmt)
      InsertTool(subtree = loop.stmt, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
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
      maxThreadNumber_node = self.getThreadNum(parallelFor.cond)


      ##################### Cuda parameters on host

      # Declarations
      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node.name)
      InsertTool(subtree = declarations_subtree, position = "end").apply(parent_stmt, 'decls')
      # Initialization
      initialization_subtree = self.buildInitializaton(shared_vars = prev_node.child.shared[0].identifiers[0].params, ast = ast)
      InsertTool(subtree = initialization_subtree, position = "begin").apply(parent_stmt, 'stmts')
      

      ##################### Cuda Kernel 

      # Kernel
      kernel_subtree = self.buildKernel(params = prev_node.child.shared[0].identifiers[0].params, 
                        private_list = prev_node.child.private[0].identifiers[0].params, 
                        reduction_list = prev_node.child.reduction[0].identifiers[0].params,
                        loop = parallelFor, ast = ast)

      # Function declaration
      # - Build a node withouth body
      import copy
      tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
      tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
      kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
      InsertTool(subtree = kernel_decl, position = "begin" ).apply(ast, 'ext')
      # Function definition
      InsertTool(subtree = kernel_subtree, position = "end" ).apply(ast, 'ext')

      # Support subtree
      support_subtree = self.buildSupport()
      InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
      InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


      ##################### Loop substitution 
      
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);
      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(prev_node.child.shared[0].identifiers[0].params[0].name)
      InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve()
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_var_node = prev_node.child.reduction[0].identifiers[0].params[0])
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
