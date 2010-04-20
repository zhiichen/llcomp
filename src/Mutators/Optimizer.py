from pycparser import c_parser, c_ast
from Visitors.generic_visitors import AttributeFilter, FilterVisitor, NodeNotFound
from Tools.tree import InsertTool, RemoveTool, ReplaceTool
from Visitors.generic_visitors import FilterVisitor, IDFilter, FuncCallFilter, FuncDeclOfNameFilter, StrFilter
from Mutators.AbstractMutator import AbstractMutator


class DeclsToParamsMutator(AbstractMutator):
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
#      self.params.show()
      InsertTool(subtree = self.params, position = "end").apply(ast, 'params')
      


class PointerMutator(AbstractMutator):
   def filter(self, ast):
     return ast

   def mutatorFunction(self, ast):
     pointer_node = c_ast.PtrDecl(type = ast.type, quals = [], parent = ast)
     ast.type.parent = pointer_node
     ast.type = pointer_node
     return ast

