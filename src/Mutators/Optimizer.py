from pycparser import c_parser, c_ast
from Visitors.generic_visitors import AttributeFilter, FilterVisitor, NodeNotFound
from Tools.tree import InsertTool, RemoveTool, ReplaceTool
from Visitors.generic_visitors import FilterVisitor, IDFilter, FuncCallFilter, FuncDeclOfNameFilter, StrFilter
from Mutators.AbstractMutator import AbstractMutator

from Visitors.generic_visitors import GenericFilterVisitor

class ConstantBinaryExpressionFilter(GenericFilterVisitor):
   """ Returns the first node with the given attribute
   """
   def __init__(self):
      super(ConstantBinaryExpressionFilter, self).__init__(condition_func = lambda node : type(node) == c_ast.BinaryOp and type(node.right) == c_ast.Constant and type(node.left) == c_ast.Constant)


#   def visit_BinaryOp(self, node, prev, offset = 1, ignore = []):
#      if type(node.left) == c_ast.Constant and type(node.right) == c_ast.Constant:
#         print "Constant Op : " + str(node.left.value) + node.op + str(node.right.value)
#         parent = node.parent
#         for attr in dir(parent):
#            if getattr(parent, attr) == node:
#               setattr(parent, attr, c_ast.Constant(value = eval(str(node.left.value) + str(node.op) + str(node.right.value)), parent = node.parent, type = node.left.type))
#               
 

class ConstantCalc(AbstractMutator):
   """ Calculate constant operations """ 
   def __init__(self):
      super(ConstantCalc, self).__init__()

   def filter(self, ast):
      """  """
      return NotImplemented

   def filter_iterator(self, ast):
      """ Fast filter , looking for binary expressions """
      return ConstantBinaryExpressionFilter().iterate(ast)

   def fast_filter(self, ast):
      """ Fast filter , looking for binary expressions """
      return ConstantBinaryExpressionFilter().dfs_iter(ast)

   def mutatorFunction(self, ast):
      """ Mutator code """
#~      print "Optimize: " + str(ast.left.value) + str(ast.op) + str(ast.right.value) + " ( " + str(ast.left.type) + " ) "
      result = c_ast.Constant(value = eval(str(ast.left.value) + str(ast.op) + str(ast.right.value)), parent = ast.parent, type = ast.left.type)
      parent = ast.parent
      # Replace the BinaryOp for a constant node
      for attr in dir(parent):
            if id(getattr(parent, attr)) == id(ast):
               setattr(parent, attr, c_ast.Constant(value = str(eval(str(ast.left.value) + str(ast.op) + str(ast.right.value))), parent = ast.parent, type = ast.left.type))
      return ast
#
#      
#
