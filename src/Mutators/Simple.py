from pycparser import c_parser, c_ast
from generic_visitors import FilterVisitor, NodeNotFound



class SimpleMutator(object):
   """ This is an example mutator, which modifies the 
      loop direction from forward (++) to backward (--) """
   def __init__(self):
      " Nothing special "
      pass

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the For node of the AST
      f = FilterVisitor(match_node_type = c_ast.For)
      node = f.apply(ast)
      return node

   def mutatorFunction(self, ast):
      """ Mutator code """
      ast.next.op = "p--" 
      

   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try:
         start_node = self.filter(ast)
         start_node.show()
         self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node
