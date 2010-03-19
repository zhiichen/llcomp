
from pycparser import c_parser, c_ast
from Visitors.generic_visitors import AttributeFilter, FilterVisitor, NodeNotFound
from Tools.tree import InsertTool, RemoveTool


class RemoveAttributeMutator(object):
   """ Remove the child of the first apperance of an attribute inside a node """
   def __init__(self, attr):
      self.attr = attr
 
   def filter(self, ast):
      af = AttributeFilter(match_attribute = self.attr)
      attr_node = af.apply(ast)
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

class DeclsToParamsMutator(object):
   """ DeclsToParams """ 
   def __init__(self, decls):
      " Save the params "
      self.params = self.convert(decls);

   def convert(self, decls):
      """ Transform a type declaration to a parameter declaration """
      # TODO: Implement this !
      # If ArrayDecl, change to PointerDecl
      # If StructDecl, change to PointerDecl
      # Else, do not touch
      # Remove initialization
      params_tmp = []
      for decl in decls:
         params_tmp += [RemoveAttributeMutator('init').apply(decl)]
      return c_ast.ParamList(params = params_tmp, coord = 0)

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
