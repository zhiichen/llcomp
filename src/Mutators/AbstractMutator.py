
from pycparser import c_parser, c_ast
from Visitors.generic_visitors import AttributeFilter, FilterVisitor, NodeNotFound
from Tools.tree import InsertTool, RemoveTool, ReplaceTool


from Visitors.generic_visitors import FilterVisitor, IDFilter, FuncCallFilter, FuncDeclOfNameFilter, StrFilter


class MutatorException(Exception):
   """ Generic mutator exception """
   def __init__(self, description):
      self.description = description 
  
   def __str__(self):
      return self.description

class IgnoreMutationException(MutatorException):
   """ Exception raised when, for some reason, we need to stop a mutation but it is not an error 
      
   """
   pass

class AbortMutationException(MutatorException):
   """ Abort mutation, error """
   pass

class AbstractMutator(object):
   def __init__(self):
      pass

   def filter(self, ast):
      pass

   def mutatorFunction(self, ast):
      return ast

   def apply(self, ast):
      start_node = None
      self.ast = ast
      try:
        start_node = self.filter(self.ast)
        self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node
 
   def filter_iterator(self, ast):
      raise NotImplemented

   def apply_all(self, ast):
      """ Apply mutation to all matches """
      start_node = None
      self.ast = ast
      try:
         for elem in self.filter_iterator(ast):
            start_node = self.mutatorFunction(elem)
      except NodeNotFound as nf:
         print str(nf)
      return start_node

   def fast_filter(self, ast):
      raise NotImplemented


   def fast_apply_all(self, ast):
      """ Apply mutation to all matches """
      start_node = None
      self.ast = ast
      for elem in self.fast_filter(ast):
        start_node = self.mutatorFunction(elem)
      return start_node


