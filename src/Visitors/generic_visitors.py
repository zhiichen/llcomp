
from pycparser import c_ast

from Tools.tree import NodeNotFound, NodeNotValid




class GenericFilterVisitor(object):
   """ Returns the first node validating a condition function
   """

   def __init__(self, condition_func, prev_brother = None):
       self.condition_func = condition_func;
       self.prev_brother = prev_brother;
       self.match = False
       self.parent_of_match = None

   def apply(self, ast):
      """ Apply filter to the ast """
      self.match = False
      node = ast
      if not self.condition_func(node):
         node = self.generic_visit(ast)
      if not self.condition_func(node):
         raise NodeNotFound("condition")
      return node

   def visit(self, node, prev, offset = 1):
        """ Visit a node. 
        """
        if self.match: return node
        if self.condition_func(node) and (self.prev_brother != None and self.prev_brother == prev):
           self.match = True
           return node
        # Continue the search....
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, offset)
        
   def generic_visit(self, node, offset = 0):
       """ Called if no explicit visitor function exists for a 
           node. Implements preorder visiting of the node.
       """
       # Store the parent node of the match
       debug = False
       iter = node.children().__iter__();
       r = node;
       c = None
       prev = None;
       if debug:
          print " Iterating the childs of node : " + str(node)
          node.show()
          print " Childs : " + str([ n for n in node.children()])
          print " Visiting " + str(node)

       try:
          c = iter.next();
          while not self.condition_func(c) or (self.prev_brother != None and self.prev_brother != prev):
             if self.match: 
                break
             if debug:
                print " Act : " + str(c)
             r = self.visit(c, prev)
             if self.condition_func(r) and (self.prev_brother != None and self.prev_brother == prev) and not self.match:
                # Stop iterating, we've found the mathing node
                # Do not execute the else code, we already have in r 
                # the matching node
                if debug: print " Act : " + str(c)
                self.match = True
                break
             prev = c
             c = iter.next()
          else:
             # r is always the matching node (even if its in the same level)
             r = c
             self.match = True
             if debug: print " Level of matching node : " + str(c) + " == " + str(c.name)
       except StopIteration:
          if debug: 
             print " Stop because : " + str(c)
             node.show()

       if debug: print " Final node : " + str(r) 
       if debug: print "==" + str(r.name)

       if (self.match == True and self.parent_of_match == None):
           self.parent_of_match = node

       return r

   def parentOfMatch(self):
       return self.parent_of_match




class FilterVisitor(GenericFilterVisitor):
   """ Returns the first node matching the node type
   """

   def __init__(self, match_node_type, prev_brother = None):
       super(FilterVisitor, self).__init__(condition_func = lambda node : type(node) == match_node_type, prev_brother = prev_brother)

class AttributeFilter(GenericFilterVisitor):
   """ Returns the first node with the given attribute
   """

   def __init__(self, match_attribute, prev_brother = None):
       super(AttributeFilter, self).__init__(condition_func = lambda node : hasattr(node, match_attribute), prev_brother = prev_brother)


class IDFilter(GenericFilterVisitor):
   """ Returns the first node with the given attribute
   """

   def __init__(self, id, prev_brother = None):
       super(IDFilter, self).__init__(condition_func = lambda node : type(node) == c_ast.ID and node.name == id.name, prev_brother = prev_brother)

