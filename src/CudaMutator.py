from pycparser import c_parser, c_ast
from generic_visitors import FilterVisitor


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
      f = FilterVisitor(match_node = c_ast.Pragma)
      node = f.generic_visit(ast)
      return node

   def mutatorFunction(self, ast):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is a Pragma
      f = FilterVisitor(match_node = c_ast.For, prev_brother = c_ast.Pragma)
      

   def apply(self, ast):
      """ Apply the mutation """
      start_node = self.filter(ast)
      start_node.show()
      self.mutatorFunction(start_node)
      return start_node
