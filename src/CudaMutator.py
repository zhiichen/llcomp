from pycparser import c_parser, c_ast
from generic_visitors import FilterVisitor, InsertVisitor, NodeNotFound

from string import Template

class CudaMutator(object):
   """ This is mutator locates a Pragma node, and then
      translate the original source to the pi cuda implementation 
     (So, it only works with CUDA...)
   """
   def __init__(self):
      " Constructor "
      self.template_parser = c_parser.CParser()

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

   def buildDeclarations(self, numThreads):
      """ Builds the declaration section 
          @param numThreads number of threads
          @return Declaration subtree, ready for insert
      """ 
      print " *** "
      print "Declaration template"
      template_code = """
      int dimA = $numThreads;
      int numThreadsPerBlock = 512;
      int numBlocks = dimA / numThreadsPerBlock;
      int memSize = numBlocks * numThreadsPerBlock * sizeof (double);
      double *reduction_loc = (double *) malloc (memSize);
      double *reduction_cu;
      """
      template_code = Template(template_code).substitute(numThreads = numThreads)
      declarations_subtree = None
      try:
         declarations_subtree = self.template_parser.parse(template_code, filename='declarations')
         print "Subtree: "
         declarations_subtree.show()
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)
      return declarations_subtree;

   def buildInitializaton(self):
      """ Initialization """
      template_code = """
      int fake() {
          cudaMalloc((void **) &reduction_cu, memSize);
      }
      """ 
      initialization_subtree = None
      parser = c_parser.CParser()
      print " Initialization subtree "
      try:
         initialization_subtree = self.template_parser.parse(template_code, filename='initialization').ext[0].body
         print "Subtree: "
         initialization_subtree.show()
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)
      return initialization_subtree

   def buildRetrieve(self):
      template_code = """
      int fake() {
      cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost);
      checkCUDAError("memcpy");
      }
      """ 
      retrieve_subtree = None
      try:
         retrieve_subtree = self.template_parser.parse(template_code, filename='Retrieve').ext[0].body
         print "Subtree: "
         retrieve_subtree.show()
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)

      return retrieve_subtree

   def buildHostReduction(self):
      template_code = """
      int fake() {
      for (i = 0; i < dimA; i++) 
      {
        sum += reduction_loc[i];
      }
      }
      """
      reduction_subtree = None
      try:
         reduction_subtree = self.template_parser.parse(template_code, filename='HostReduction').ext[0].body
         print "Subtree: "
         reduction_subtree.show()
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)
      return reduction_subtree




   def mutatorFunction(self, ast, prev_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
      filter = FilterVisitor(match_node_type = c_ast.For, prev_brother = prev_node)
      parallelFor = filter.apply(ast)
      print " Found : "
      parallelFor.show()
      # Parent of the node
      parent_stmt = filter.parentOfMatch()
      print "Parent "
      parent_stmt.show()
      print "Number of threads:"
      maxThreadNumber_node = self.getThreadNum(parallelFor.cond)
      declarations_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node.name)
      InsertVisitor(subtree = declarations_subtree, position = "end").apply(parent_stmt, 'decls')
      initialization_subtree = self.buildInitializaton()
      InsertVisitor(subtree = initialization_subtree, position = "begin").apply(parent_stmt, 'stmts')
      retrieve_subtree = self.buildRetrieve()
      InsertVisitor(subtree = retrieve_subtree, position = "end").apply(parent_stmt, 'stmts')
      reduction_subtree = self.buildHostReduction()
      InsertVisitor(subtree = reduction_subtree, position = "end").apply(parent_stmt, 'stmts')


   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try: 
         print " Searching pragma "
         start_node = self.filter(ast)
         print "Pragma found: "
         start_node.show()
         print " >>> Mutating tree <<<<"
         self.mutatorFunction(ast, start_node)
      except NodeNotFound as nf:
         print nf
      return ast
