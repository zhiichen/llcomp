from pycparser import c_ast

import sys


class DotWriter(object):
   inside = False

   def __init__(self, filename = None, highlight = None):
      self.act_id = 0
      self.highlight = highlight
      self.node_dir = { }
      self.show_types = True
      self.filename = filename or sys.stdout
      try:
         self.file = open(self.filename, 'w+') 
      except TypeError:
         self.file = sys.stdout

      self.inside = False
      
      self.writeLn(0, "digraph G {")
      if self.highlight:
         self.write_highlights()

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
      self.writeLn(0, begin  + "-> " + end + "[label = \"" + name + "\"]" + ";")

   def write_label(self, node, name):
      self.writeLn(0, node + "[label = \"" + name + "\"]" + ";")

   def write_highlights(self):
      for node in self.highlight:
         dot_name = self.get_name(node)
         self.writeLn(0, dot_name +" [shape=box, color=red, style=filled];")

   def get_name(self, node):
      """ Get the name of a node """
      if node in self.node_dir:
         return self.node_dir[node]


      if type(node) == type(""):
         dot_name = node
         self.node_dir[node] = node

      elif 'name' in dir(node) and getattr(node, 'name'):
         if type(node.name) != type(""):
            dot_name = self.get_name(node.name) 
            self.write_label(dot_name, dot_name.split('__DOT__')[0])
         elif not node.name in self.node_dir:
            dot_name = node.name + "_" + str(self.act_id)
            self.write_label(dot_name, node.name)
      elif 'names' in dir(node) and getattr(node, 'names'):
            if len(node.names) == 1 and node.names[0] in self.node_dir:
               dot_name = node.names[0]
            else:
               dot_name = "_".join(node.names) + "_" + str(self.act_id)
               self.write_label(dot_name, " ".join(node.names))
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
            if isinstance(getattr(node, attr), c_ast.TypeDecl) and attr != "parent":
               if self.show_types:
                 self.write_line(dot_name, attr, self.get_name(getattr(node, attr)))
                 self.visit(getattr(node, attr))
            else:
               self.write_line(dot_name, attr, self.get_name(getattr(node, attr)))
               self.visit(getattr(node, attr))
         if type(getattr(node, attr)) == type([]):
            for elem in getattr(node, attr):
               # Avoid cycles
               if dot_name != self.get_name(elem):
                  self.write_line(dot_name, attr, self.get_name(elem))
               self.visit(elem, offset)
         

            
#      for c in node.children():
#         self.visit(node, c)




