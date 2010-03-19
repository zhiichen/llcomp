from pycparser import c_parser, c_ast
from Visitors.generic_visitors import FilterVisitor
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Mutators.DeclsToParams import DeclsToParamsMutator

from string import Template


class KernelLoopVarMutator(object):
   """ 1. Replace the loop var  to idx
           -> Filter by ID node
           -> change name to idx
   """
   def __init__(self, loop_var):
      self.loop_var = loop_var
 
   def filter(self, ast):
      af = FilterVisitor(match_node_type = c_ast.ID)
      id_node = af.apply(ast)
      return [attr_node, af.parentOfMatch()]

   def mutatorFunction(self, ast, parent):
      del ast.init
      ast.init = None
      return ast

   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try:
         [start_node, parent] = self.filter(ast)
         self.mutatorFunction(start_node, parent)
      except NodeNotFound as nf:
         print str(nf)
      return start_node


class CudaMutator(object):
   """ This is mutator locates a Pragma node, and then
      translate the original source to the pi cuda implementation 
   """
   def __init__(self):
      " Constructor "
      # BUG: Don't work with optimize
      self.template_parser = c_parser.CParser(lex_optimize = False, yacc_optimize = False)

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
      import subprocess
      from cStringIO import StringIO
      try:
         p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
         clean_source = p.communicate(template_code)[0]
         process = subprocess.Popen("sed -nf nocomments.sed", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
         stripped_code = process.communicate(clean_source)[0]
         subtree = self.template_parser.parse(stripped_code, filename=name)
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)
	#  print " Code : " + str(stripped_code)
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
#define  __attribute__(x)  /*NOTHING*/


#define __const
#define __addr
#define __THROW
#define __extension__

# define __inline
# define __THROW
# define __P(args)   args
# define __PMT(args) args
# define __restrict__
# define __restrict


#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/builtin_types.h"

       int fake() {
	    dim3 dimGrid (numBlocks);
   	    dim3 dimBlock (numThreadsPerBlock);
 	    piLoop <<< dimGrid , dimBlock >>> (reduction_cu, $sharedvars);
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, {'sharedvars' : shared_var_list}, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
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


   def buildKernel(self, params, private_list, reduction_list, loop, ast):
      if not Dump.exists('KernelBuild'):
          template_code = """
          __global__ void piLoop (double * reduction_cu)
          {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          reduction_cu[idx]  = 1.0;
          }
          """
          print " *** Parse *** "
          tree = self.parse_snippet(template_code, None, name = 'KernelBuild')
          Dump.save('KernelBuild', tree)
      else:
          print " *** Load *** "
          tree = Dump.load('KernelBuild')
      # OpenMP shared vars are parameters of the kernel function
      # Note: we need to mutate the declaration subtree into a param declaration (ArrayRef to Pointer and so on...)
      param_decls = [ decl_of_id(elem, ast) for elem in params ]
      pm = DeclsToParamsMutator(decls = param_decls)
      pm.apply(tree.ext[0].function.decl.type.args)
      # OpenMP Private vars need to be declared inside kernel
      #    - we build a tmp Compound to group all declarations, and insert them once
      tmp = c_ast.Compound(decls= [decl_of_id(elem, ast) for elem in private_list], stmts=None)
      #    - Insert tool removes the parent node of the inserted subtree
      InsertTool(subtree = tmp, position = "end").apply(tree.ext[0].function.body, 'decls')
      # Add the loop statements, (but not the reduction)
 #     km = KernelMutator(loop_var = dec_of_id(reduction_list[0]))
 #     InsertTool(subtree = kernel_stmts, position = "begin").apply(tree.ext[0].function.body, 'stmts')
      return tree

   def mutatorFunction(self, ast, prev_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
      filter = FilterVisitor(match_node_type = c_ast.For, prev_brother = prev_node)
      parallelFor = filter.apply(ast)
      # Parent of the node
      parent_stmt = filter.parentOfMatch()
      maxThreadNumber_node = self.getThreadNum(parallelFor.cond)
      # Build subtrees
      # Kernel
      kernel_subtree = self.buildKernel(params = prev_node.child.shared[0].identifiers[0].params, 
			private_list = prev_node.child.private[0].identifiers[0].params, 
                        reduction_list = prev_node.child.reduction[0].identifiers[0].params,
                        loop = parallelFor, ast = ast)
      InsertTool(subtree = kernel_subtree, position = "end").apply(ast, 'ext')
      # Declarations
      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node.name)
      InsertTool(subtree = declarations_subtree, position = "end").apply(parent_stmt, 'decls')
      # Initialization
      initialization_subtree = self.buildInitializaton(shared_vars = prev_node.child.shared[0].identifiers[0].params, ast = ast)
      InsertTool(subtree = initialization_subtree, position = "begin").apply(parent_stmt, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve()
      InsertTool(subtree = retrieve_subtree, position = "end").apply(parent_stmt, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_var_node = prev_node.child.reduction[0].identifiers[0].params[0])
      InsertTool(subtree = reduction_subtree, position = "end").apply(parent_stmt, 'stmts')
      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(prev_node.child.shared[0].identifiers[0].params[0].name)
      ReplaceTool(new_node = kernelLaunch_subtree, old_node = parallelFor).apply(parent_stmt, 'stmts')
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
