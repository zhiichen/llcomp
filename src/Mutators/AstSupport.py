
from pycparser import c_parser, c_ast
from Visitors.generic_visitors import AttributeFilter, FilterVisitor, NodeNotFound
from Tools.tree import InsertTool, RemoveTool, ReplaceTool

from Visitors.generic_visitors import FilterVisitor, IDFilter, FuncCallFilter, FuncDeclOfNameFilter, StrFilter, FilterError

from Mutators.AbstractMutator import AbstractMutator, IgnoreMutationException, AbortMutationException

from Tools.search import type_of_id, decl_of_id


class RemoveAttributeMutator(AbstractMutator):
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

class DeclsToParamsMutator(AbstractMutator):
   """ DeclsToParams """ 
   def __init__(self):
      " Save the params "
      # self.params = self.convert(decls);

   def convert(self, node):
      """ Transform a type declaration to a parameter declaration """
      # TODO: Implement this !
      # If ArrayDecl, change to PointerDecl DONE
      # If StructDecl, change to PointerDecl
      # Else, do not touch
      # Remove initialization DONE
      params_tmp = []
      decls = node.params
      for decl in decls:
         new_decl = RemoveAttributeMutator('init').apply(decl);
         if isinstance(new_decl.type, c_ast.ArrayDecl) or isinstance(new_decl.type, c_ast.Struct):
            PointerMutator().apply(new_decl)
         params_tmp.append(new_decl) 

      return c_ast.ParamList(params = params_tmp, coord = 0, parent = node.parent)

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter
      """
      # Build a visitor , matching the ParamList node of the AST
      f = FilterVisitor(match_node_type = c_ast.ParamList)
      node = f.apply(ast)
      return node

   def mutatorFunction(self, ast):
      """ Mutator code """
#      InsertTool(subtree = self.params, position = "end").apply(ast, 'params')
      # ReplaceTool(new_node = self.params, old_node = self.params).apply(ast, 'params')
      ast.params = self.convert(ast).params


class IDNameMutator(AbstractMutator):
   """  Replace and ID name with another ID name """
   def __init__(self, old, new):
      self.old = old
      self.new = new
 
   def filter(self, ast):
      id_node = None
      try:
         af = IDFilter(id = self.old)
         id_node = af.apply(ast)
      except NodeNotFound:
         # Try to recover looking for No ID name attribute
         af = StrFilter(id = self.old)
         id_node = af.apply(ast)
         return id_node
         # Otherwise , raise exception and stop
      return id_node

   def filter_iterator(self, ast):
      id_node = None
      af = IDFilter(id = self.old)
#~      print " IDNameMutator iterator ast: " + str(ast) + "Looking for : " + str(self.old.name)
      try:
         for id_node in af.iterate(ast):
#~            print "ID Node : " + str(id_node)
            yield id_node
         else:
            af = StrFilter(id = self.old)
            for id_node in af.iterate(ast):
#~               print "Str Node  : " + str(id_node)
               yield id_node
            # Otherwise , raise exception and stop
            raise StopIteration
      except NodeNotFound:
         # Try to recover looking for No ID name attribute
         af = StrFilter(id = self.old)
         for id_node in af.iterate(ast):
            yield id_node
         # Otherwise , raise exception and stop
         raise StopIteration

      print " You shouldn't see this " 
#      return id_node


   def mutatorFunction(self, ast):
      if hasattr(ast, 'name'):
         delattr(ast, 'name')
         setattr(ast, 'name', self.new.name)
      elif hasattr(ast, 'declname'):
         delattr(ast, 'declname')
         setattr(ast, 'declname', self.new.name)
      return ast




class FuncToDeviceMutator(AbstractMutator):
   """  Replace the definition of a FuncCall with a CUDAKernel with type __device__ """
   def __init__(self, func_call):
      self.func_call = func_call
      self.ignore_names = ['sqrt', 'min', 'max', 'log', 'fabs']
 
   def filter(self, ast):
      """ Find the declaration of the """
      id_node = None
      if self.func_call.name.name in self.ignore_names:
         raise IgnoreMutationException('This function is already implemented on CUDA')
      try:
         # ast.show()
 #        from Tools.Debug import DotDebugTool
 #        DotDebugTool().apply(self.func_call)
         decl = decl_of_id(self.func_call.name, ast)
         if hasattr(decl, 'storage') and 'extern' in decl.storage:
#            print "********************"
#            print dir(decl)
#            decl.show()
#            print "********************"
            print " *** Cannot use external declarations inside kernel *** "
            print " Decl : " + decl.name
            raise FilterError("Cannot use external declarations inside kernel")
         af = FuncDeclOfNameFilter(name = self.func_call.name)
         id_node = af.apply(ast)
#         print " Declaration " + str(self.func_call.name) + " is " + str(id_node) 
#         print " Definition " + str(id_node.parent.parent)
      except NodeNotFound:
         print " *** Node not found *** "
         return None
#      if isinstance(id_node.parent, c_ast.Decl):
      if isinstance(id_node.parent.parent, c_ast.FileAST):
         for elem in id_node.parent.parent.ext:
            if isinstance(elem, c_ast.FuncDef) and elem.decl.name == self.func_call.name:
               return elem
         raise NodeNotFound(id_node)
      else:
         return id_node.parent.parent

   def mutatorFunction(self, ast):
      file_ast = ast
      while type(file_ast) != c_ast.FileAST:
         file_ast = file_ast.parent

      cuda_node = c_ast.CUDAKernel(name = self.func_call.name.name, type = 'both', function = ast, parent=file_ast)
      ast.parent = cuda_node
#      from Tools.Debug import DotDebugTool
#      DotDebugTool(highlight = [ast]).apply(file_ast)
#      ast.show()
      ReplaceTool(new_node = cuda_node, old_node = ast).apply(file_ast, 'ext')
      return cuda_node



class PointerMutator(AbstractMutator):
   def filter(self, ast):
     return ast

   def mutatorFunction(self, ast):
     if not isinstance(ast.type, c_ast.ArrayDecl):
        pointer_node = c_ast.PtrDecl(type = ast.type, quals = [], parent = ast)
        ast.type.parent = pointer_node
        ast.type = pointer_node
     else:
        pointer_node = c_ast.PtrDecl(type = ast.type.type, quals = [], parent = ast)
        ast.type.parent = pointer_node
        ast.type = pointer_node

     return ast



