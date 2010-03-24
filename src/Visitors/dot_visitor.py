from pycparser import c_parser, c_ast

import sys


class DotWriter(object):
   inside = False

   def __init__(self, filename = None):
      self.act_id = 0
      self.node_dir = { }
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


   # ********** Dot language **********

   def write_line(self, begin, name, end):
      self.write(0, begin  + "-> " + end + "[label = \"" + name + "\"]" + ";")

   def write_label(self, node, name):
      self.write(0, node + "[label = \"" + name + "\"]" + ";")


   def get_name(self, node):
      """ Get the name of a node """
      if node in self.node_dir:
         return self.node_dir[node]
      if 'name' in dir(node) and getattr(node, 'name'):
         if type(node.name) != type(""):
            dot_name = self.get_name(node.name) # node.name.name + "_" + str(self.act_id)
            self.write_label(dot_name, dot_name.split('__DOT__')[0])
         else:
            dot_name = node.name + "_" + str(self.act_id)
            self.write_label(dot_name, node.name)
      else:
         dot_name = node.__class__.__name__ + "__DOT__" + str(self.act_id)
         self.write_label(dot_name, node.__class__.__name__)
      self.act_id = self.act_id + 1
      self.node_dir[node] = dot_name
      return dot_name


   # ********** Visit **********

   def visit(self, node, offset = 0):
      """ Visit a node. 
      """
      method = 'visit_' + node.__class__.__name__
      visitor = getattr(self, method, self.generic_visit)
      return visitor(node, offset)

   def generic_visit(self, node, offset = 0):
      """ Called if no explicit visitor function exists for a 
         node. Implements preorder visiting of the node.
       """
      label = ""
      dot_name = self.get_name(node)
      for attr in dir(node):
         if isinstance(getattr(node, attr), c_ast.Node) and attr != "parent":
            self.write_line(dot_name, attr, self.get_name(getattr(node, attr)))
            self.visit(getattr(node, attr))
         if type(getattr(node, attr)) == type([]):
            for elem in getattr(node, attr):
               self.write_line(dot_name, attr, self.get_name(elem))
               self.visit(elem, offset)
         

            
#      for c in node.children():
#         self.visit(node, c)




