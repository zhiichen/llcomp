"""Tools to manage the internal representation of the code


.. moduleauthor:: Ruyman Reyes Castro <rreyes@ull.es>

"""

from pycparser import  c_ast

import cStringIO


class AstToIR:
   """Transform a C ast to the internal representation

   """
   def __init__(self, Writer):
      def fast_write(node):
         """Write the internal representation with the given Writer
            :param node: Node to write
         """
         writeIO = cStringIO.StringIO()
         cw = Writer(stream = writeIO)
         cw.visit(node)
         return writeIO.getvalue()

      # Overload __str__ method on Node class
      # Will apply to all instances !!
      c_ast.Node.__str__ = fast_write

   
   def deep_first_search(self, root, visited = None, preorder_process  = lambda x: None):
      """Given a starting vertex, root, do a depth-first search.

         :param root: Root to start search
         :param visited: List of already visited nodes
         :param preorder_process: Function to apply in all nodes
         
      """
      to_visit = [] 
      if visited is None: visited = set()
   
      to_visit.append(root) # Start with root
      while len(to_visit) != 0:
              v = to_visit.pop()
              if v not in visited:
                      visited.add(v)
                      preorder_process(v)
                      to_visit.extend(v.children())

   def link_all_parents(self, ast):
        """ Function to link the nodes of the AST in reverse order, using a parent attribute in each node 

            :param ast: Ast to relink
        """
        def link_parent(node):
              for child in node.children():
                      child.parent = node

        self.deep_first_search(root = ast, visited = None, preorder_process = link_parent)

   def transform(self, node):
      """ Apply transformations needed to migrate from a c_ast to a InternalRepresentation 
         
         :param node: FileAST node to start conversion
      """
      self.link_all_parents(node)  
      return node

   def update(self, node):
      """ Update parent links
         
         :param node: FileAST node to start conversion
      """
      self.link_all_parents(node)  
      return node


   def symbolTable(self,node):
      """ Returns the symbol table for the given AST
         
         :param node: FileAST node to start conversion
      """
      raise NotImplemented
