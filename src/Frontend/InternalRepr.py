from pycparser import parse_file, c_ast
from Visitors.clone_visitor import CWriter, OmpWriter
import cStringIO


class AstToIR:
   """ Transform a C ast to the internal representation """

   def __init__(self, writer):
      def fast_write(node):
         writeIO = cStringIO.StringIO()
         cw = writer(stream = writeIO)
         cw.visit(node)
         return writeIO.getvalue()

      # Overload __str__ method on Node class
      # Will apply to all instances !!
      c_ast.Node.__str__ = fast_write

   
   def deep_first_search(self, root, visited = None, preorder_process  = lambda x: None):
      """
       Given a starting vertex, root, do a depth-first search.
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
        """ Function to link the nodes of the AST in reverse order, using a parent attribute in each node """
        def link_parent(node):
              for child in node.children():
                      child.parent = node

        self.deep_first_search(root = ast, visited = None, preorder_process = link_parent)

   def transform(self, node):
      self.link_all_parents(node)  
      return node

   def symbolTable(self,node):
      raise NotImplemented
