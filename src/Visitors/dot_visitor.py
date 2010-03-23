from pycparser import c_parser, c_ast

import sys


class DotWriter(object):
   inside = False

   def __init__(self, filename = None):
      self.filename = filename or sys.stdout
      try:
         self.file = open(self.filename, 'w+') 
      except TypeError:
         self.file = sys.stdout

      self.inside = False
      
      self.writeLn(0, "digraph G {")

   def __del__(self):
      """ Ensure closing the file when object dissapears """
      self.writeLn(0, "}")
      self.file.close()


   # ********** Writing support **********

   def writeLn(self, offset, string):
      self.write(offset, string)
      self.file.write("\n")

   def write(self, offset, string):
      self.file.write(" " * offset + string)

   def debug(self, node):
      print " >>>>> * <<<< "
      node.show()
      print " >>>> " + str(dir(node)) + "<<< "
      print " >>>>> * <<<< "

   def write_blank(self):
      self.file.write(" ")


   def visit(self, node, offset = 0):
      """ Visit a node. 
      """
      method = 'visit_' + node.__class__.__name__
      visitor = getattr(self, method, self.generic_visit)
      visit = False
      if type(node) == c_ast.FileAST:
         self.writeLn(0, "start ->" + node.__class__.__name__ + ";")
         visit = True
      else:
         if hasattr(node.parent, 'name') and node.parent.name:
            if type(node.parent.name) == c_ast.ID:
               self.write (0, node.parent.name.name)
            else:
               self.write (0, str(node.parent.name))
         else:
            self.write (0, node.parent.__class__.__name__)
         self.write(0, "->")
         if hasattr(node, 'name') and node.name:
            if type(node.name) == c_ast.ID:
               self.write (0, node.name.name)
            else:
               self.write (0, str(node.name))
               visit = True
         else:
            self.write(0, node.__class__.__name__)
            visit = True 
         self.writeLn(0, ";")
      if visit:
         return visitor(node, offset)
      else:
         return None

   def generic_visit(self, node, offset = 0):
      """ Called if no explicit visitor function exists for a 
         node. Implements preorder visiting of the node.
       """
      for c in node.children():
         self.visit(c)



