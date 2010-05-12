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
      def condition(node):
         if type(node) == c_ast.BinaryOp and type(node.right) == c_ast.Constant and type(node.left) == c_ast.Constant:
            return True
         elif type(node) == c_ast.Constant and len(node.value) > 1:
            return True
         return False
      super(ConstantBinaryExpressionFilter, self).__init__(condition_func = condition)


class ConstantCalc(AbstractMutator):
   """ Calculate constant operations """ 
   def __init__(self):
      super(ConstantCalc, self).__init__()

   def filter(self, ast):
      """  """
      raise NotImplemented

   def filter_iterator(self, ast):
      """ Fast filter , looking for binary expressions """
      return ConstantBinaryExpressionFilter().iterate(ast)

   def fast_filter(self, ast):
      """ Fast filter , looking for binary expressions """
      return ConstantBinaryExpressionFilter().dfs_iter(ast)

   def mutatorFunction(self, ast):
      """ Mutator code """
      result = None
      _type = None
      if type(ast) == c_ast.BinaryOp:
#~      print "Optimize: " + str(ast.left.value) + str(ast.op) + str(ast.right.value) + " ( " + str(ast.left.type) + " ) "
         result = c_ast.Constant(value = str(eval(str(ast.left.value) + str(ast.op) + str(ast.right.value))), parent = ast.parent, type = ast.left.type)
         _type =  ast.left.type
      elif type(ast) == c_ast.Constant:
         if ast.type == 'int':
            try:
               result = c_ast.Constant(value = str(eval(ast.value)), parent = ast.parent, type = 'int')
            except NameError:
               print "Name error: " + str(ast.value)
               result = c_ast.Constant(value = str(ast.value), parent = ast.parent, type = 'int')
         else:
            result = c_ast.Constant(value = str(ast.value), parent = ast.parent, type = 'string')


      parent = ast.parent
      # Replace the BinaryOp for a constant node
      for attr in dir(parent):
         if id(getattr(parent, attr)) == id(ast):
            setattr(parent, attr, result)
      return ast


class MatrixDeclFilter(GenericFilterVisitor):
   """ Returns the first node with the given attribute
   """
   def __init__(self):
      super(MatrixDeclFilter, self).__init__(condition_func = lambda node : type(node) == c_ast.ArrayDecl 
                                                   and type(node.type) == c_ast.ArrayDecl 
                                                   and not type(node.type.type) == c_ast.ArrayDecl)


class MatrixDeclToPtr(AbstractMutator):
   """ Convert a Matrix Declaration to a dynamic vector """
   def __init__(self, start_ast):
      self.nrows = None
      self.ncols = None
      self.start_ast = start_ast
      super(MatrixDeclToPtr, self).__init__()

   def filter(self, ast):
      """ """
      raise NotImplemented

   def filter_iterator(self, ast):
      """ Iterable """
      return MatrixDeclFilter().iterate(ast)


   def fast_filter(self, ast):
      """ """
      return MatrixDeclFilter().dfs_iter(ast)
  
   def mutatorFunction(self, ast):
      """ Mutator code """
      array1lvl = ast
      array2lvl = ast.type
      # Ensure we're working with a matrix
      assert type(array2lvl) == c_ast.ArrayDecl

      self.nrows = array1lvl.dim.value
      self.ncols = array2lvl.dim.value

      newdim = str(array1lvl.dim.value) + " * " + str(array2lvl.dim.value)
      array1lvl.type = array2lvl.type
      array1lvl.dim = c_ast.Constant(type = 'int', value = newdim);

      MatrixRefToVect(nrows = self.nrows, ncols = self.ncols, arrayDeclName = array1lvl.parent.name).fast_apply_all(self.start_ast)      

class MatrixRefFilter(GenericFilterVisitor):
   def __init__(self, idname):
      def condition(node):
         if type(node) == c_ast.ArrayRef and type(node.name) == c_ast.ArrayRef:
            node.show()
            if node.name.name.name == idname:
               return True
         return False
      super(MatrixRefFilter, self).__init__(condition_func = condition)



# TODO: Check if this is correct
class MatrixRefToVect(AbstractMutator):
   def __init__(self, nrows, ncols, arrayDeclName):
      self.nrows = nrows
      self.ncols = ncols
      self.arrayDeclName = arrayDeclName
      super(MatrixRefToVect, self).__init__()

   def filter(self, ast):
      """ """
      raise NotImplemented

   def filter_iterator(self, ast):
      """ Iterable """
      return MatrixRefFilter(idname = self.arrayDeclName).iterate(ast)


   def fast_filter(self, ast):
      """ """
      return MatrixRefFilter(idname = self.arrayDeclName).dfs_iter(ast)
  
   def mutatorFunction(self, ast):
      """ Mutator code """
      def get_str(node):
         if hasattr(node, 'name'):
            return node.name
         elif hasattr(node, 'value'):
            return node.value
         elif type(node) == c_ast.ID:
            return node.name.name

      array1lvl = ast
      array2lvl = ast.name
      # Ensure we're working with a matrix
      assert type(array2lvl) == c_ast.ArrayRef

      i = j = None

      if type(array1lvl.subscript) == c_ast.BinaryOp:
         i = get_str(array1lvl.subscript.left) + array1lvl.subscript.op + get_str(array1lvl.subscript.right)
      else:
         i = get_str(array1lvl.subscript)
 
      if type(array2lvl.subscript) == c_ast.BinaryOp:
         j = get_str(array2lvl.subscript.left) + array2lvl.subscript.op + get_str(array2lvl.subscript.right)
      else:
         j = get_str(array2lvl.subscript)


      # i*M+j
#      newsubscript = c_ast.BinaryOp(right = c_ast.BinaryOp(left=c_ast.Constant(type='int', value=str(i)), op = " * ", right=c_ast.Constant(type='int', value=str(self.ncols))), op = " + ", left = c_ast.Constant(type='int', value=str(j)))
      newsubscript = str(i) + "*" + str(self.ncols) + "+" + str(j)

      array1lvl.name = array2lvl.name
      array1lvl.subscript = c_ast.Constant(type = 'int', value = newsubscript);








