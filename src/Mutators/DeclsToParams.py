
from pycparser import c_parser, c_ast
from Visitors.generic_visitors import FilterVisitor, NodeNotFound
from Tools.tree import InsertTool


class DeclsToParamsMutator(object):
   """ DeclsToParams """ 
   def __init__(self, decls):
      " Save the params "
      self.params = self.convert(decls);

   def convert(self, decls):
      """ Transform a type declaration to a parameter declaration """
      print "****"
      print decls
      return c_ast.ParamList(params = decls, coord = 0)

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the ParamList node of the AST
      f = FilterVisitor(match_node_type = c_ast.ParamList)
      node = f.apply(ast)
      return node

   def mutatorFunction(self, ast):
      """ Mutator code """
      self.params.show()
      InsertTool(subtree = self.params, position = "end").apply(ast, 'params')
      

   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try:
         start_node = self.filter(ast)
         self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node
