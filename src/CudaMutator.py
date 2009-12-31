from pycparser import c_parser, c_ast
from generic_visitors import FilterVisitor, NodeNotFound


class CudaMutator(object):
   """ This is mutator locates a Pragma node, and then
      translate the original source to the pi cuda implementation 
     (So, it only works with CUDA...)
   """
   def __init__(self):
      " Nothing special "
      pass

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the Pragma node of the AST
      f = FilterVisitor(match_node_type = c_ast.Pragma)
      node = f.generic_visit(ast)
      return node

   def mutatorFunction(self, ast, prev_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
      f = FilterVisitor(match_node_type = c_ast.For, prev_brother = prev_node)
      f.generic_visit(ast)
      

   def apply(self, ast):
      """ Apply the mutation """
      print " Searching node "
      start_node = self.filter(ast)
      print "Matched node: "
      start_node.show()
      print " >>> Mutating tree <<<<"
#      try: 
#         self.mutatorFunction(ast, start_node)
#      except NodeNotFound as nf:
#         print nf
      return start_node
