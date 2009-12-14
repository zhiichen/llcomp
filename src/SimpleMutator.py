from pycparser import c_parser, c_ast


class FilterVisitor(object):

   def __init__(self, match_node):
       self.match_node = match_node;

   def visit(self, node, offset = 1):
        """ Visit a node. 
        """
        print " Visiting: " + node.__class__.__name__
        if type(node) == self.match_node:
           return node
        # Continue the search....
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, offset)
        
   def generic_visit(self, node, offset = 0):
       """ Called if no explicit visitor function exists for a 
           node. Implements preorder visiting of the node.
       """
       iter = node.children().__iter__();
       r = node;
       try:
          c = iter.next();
          while type(c) != self.match_node:
             r = self.visit(c)
             if type(r) == self.match_node:
                # Stop iterating, we've found the mathing node
                # Do not execute the else code, we already have in r 
                # the matching node
                break
             c = iter.next()
          else:
             # r is always the matching node (even if its in the same level)
             r = c
       except StopIteration:
          pass
       return r


class SimpleMutator(object):
   """ This is an example mutator, which modifies the 
      loop direction from forward (++) to backward (--) """
   def __init__(self):
      " Nothing special "
      pass

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the For node of the AST
      f = FilterVisitor(match_node = c_ast.For)
      node = f.generic_visit(ast)
      return node

   def mutatorFunction(self, ast):
      """ Mutator code """
      ast.next.op = "p--" 
      

   def apply(self, ast):
      """ Apply the mutation """
      start_node = self.filter(ast)
      start_node.show()
      self.mutatorFunction(start_node)
      return start_node
