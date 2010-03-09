from pycparser import c_parser, c_ast
from generic_visitors import FilterVisitor, InsertTool, NodeNotFound, ReplaceTool, RemoveTool

from string import Template

class CudaMutator(object):
   """ This is mutator locates a Pragma node, and then
      translate the original source to the pi cuda implementation 
   """
   def __init__(self):
      " Constructor "
      # BUG: Don't work without optimize
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
      except IOError:
         print "Pipe Error"
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

   def buildInitializaton(self):
      """ Initialization """
      template_code = """
      int fake() {
          cudaMalloc((void **) &reduction_cu, memSize);
      }
      """ 
      return self.parse_snippet(template_code, None, name = 'SendData').ext[0].body

   def buildRetrieve(self):
      template_code = """
      int fake() {
      cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost);
      checkCUDAError("memcpy");
      }
      """ 
      return self.parse_snippet(template_code, None, name = 'Retrieve').ext[0].body
      
   def buildKernelLaunch(self):
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
 	 		 piLoop <<< dimGrid , dimBlock >>> (reduction_cu);
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, None, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
       return tree


   def buildHostReduction(self):
      template_code = """
      int fake() {
      for (i = 0; i < dimA; i++)
      {
        sum += reduction_loc[i];
      }
      }
      """
      return self.parse_snippet(template_code, None, name = 'HostReduction').ext[0].body


   def buildKernel(self, params):
      template_code = """
      #define __global__
      int idx;
      __global__ piLoop (double * reduction_cu, $params)
      {
      int idx = blockIdx.x*blockDim.x + threadIdx.x;
      double x = h * ((double)idx - 0.5);
      reduction_cu[idx]  = 4.0 / (1.0 + x * x);
      }
      """
      tree = self.parse_snippet(template_code, {'params' : params}, name = 'KernelBuild').ext[1]
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
      params = "double h";
      kernel_subtree = self.buildKernel(params)
      InsertTool(subtree = kernel_subtree, position = "end").apply(ast, 'ext')
      # Declarations
      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node.name)
      InsertTool(subtree = declarations_subtree, position = "end").apply(parent_stmt, 'decls')
      # Initialization
      initialization_subtree = self.buildInitializaton()
      InsertTool(subtree = initialization_subtree, position = "begin").apply(parent_stmt, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve()
      InsertTool(subtree = retrieve_subtree, position = "end").apply(parent_stmt, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction()
      InsertTool(subtree = reduction_subtree, position = "end").apply(parent_stmt, 'stmts')
      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch()
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
