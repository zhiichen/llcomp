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
#~      print "Optimize: " + str(ast.left.value) + str(ast.op) + str(ast.right.value) + " ( " + str(ast.left.type) + " ) "
      result = c_ast.Constant(value = eval(str(ast.left.value) + str(ast.op) + str(ast.right.value)), parent = ast.parent, type = ast.left.type)
      parent = ast.parent
      # Replace the BinaryOp for a constant node
      for attr in dir(parent):
            if id(getattr(parent, attr)) == id(ast):
               setattr(parent, attr, c_ast.Constant(value = str(eval(str(ast.left.value) + str(ast.op) + str(ast.right.value))), parent = ast.parent, type = ast.left.type))
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
            print "*** (name : " + str(idname)
            print "** ( : " + str(node.name.name.name)
            node.show()
            if node.name.name.name == idname:
               print " Match ! "
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
      array1lvl = ast
      array2lvl = ast.name
      # Ensure we're working with a matrix
      assert type(array2lvl) == c_ast.ArrayRef

      i = j = None

      if hasattr(array1lvl.subscript, 'name'):
         i = array1lvl.subscript.name
      else:
         i = array1lvl.subscript.value
 
      if hasattr(array2lvl.subscript, 'name'):
         j = array2lvl.subscript.name
      else:
         j = array2lvl.subscript.value


      # i*M+j
#      newsubscript = c_ast.BinaryOp(right = c_ast.BinaryOp(left=c_ast.Constant(type='int', value=str(i)), op = " * ", right=c_ast.Constant(type='int', value=str(self.ncols))), op = " + ", left = c_ast.Constant(type='int', value=str(j)))
      newsubscript = str(i) + "*" + str(self.ncols) + "+" + str(j)

      array1lvl.name = array2lvl.name
      array1lvl.subscript = c_ast.Constant(type = 'int', value = newsubscript);








