
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

   def apply(self, ast, ignore = []):
      """ Apply filter to the ast """
      self.match = False
      self.ast = ast
      node = ast
      if not self.condition_func(node):
         node = self.generic_visit(ast, ignore = ignore)
      if not self.condition_func(node):
         raise NodeNotFound(self.condition_func.__doc__)
      return node

   def visit(self, node, prev, offset = 1, ignore = []):
        """ Visit a node. 
        """
        # Continue the search....
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, offset, ignore)
        
   def generic_visit(self, node, offset = 0, ignore = []):
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
          while not self.condition_func(c) or (self.prev_brother != None and self.prev_brother != prev) or (c in ignore):
             if self.match:
               break
             if debug:
                print " Act : " + str(c)
             r = self.visit(c, prev, ignore = ignore)
             if self.condition_func(r) and (self.prev_brother != None and self.prev_brother == prev) and not self.match and not r in ignore:
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
             if not c in ignore:
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

   def iterate(self, ast):
       """ Iterate through matching nodes """
       visited_nodes = []
       try:
         while 1:
            visited_nodes.append(self.apply(ast, ignore = visited_nodes))
            yield visited_nodes[-1]
       except NodeNotFound:
#         print "   *** Node not found on iterate, will raise StopIteration *** "
#         raise NodeNotFound("Not")
         raise StopIteration




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
   """ Returns the first node with an ID
   """

   def __init__(self, id, prev_brother = None):
       super(IDFilter, self).__init__(condition_func = lambda node : type(node) == c_ast.ID and node.name == id.name, prev_brother = prev_brother)

class StrFilter(GenericFilterVisitor):
   """ Returns the first node with an ID
   """

   def __init__(self, id, prev_brother = None):
       super(StrFilter, self).__init__(condition_func = lambda node : hasattr(node, 'name') and type(node.name) == type("") and node.name == id.name, prev_brother = prev_brother)


class DeclFilter(GenericFilterVisitor):
   """ Returns the first node with a TypeDecl
   """

   def __init__(self, attribute, value, prev_brother = None):
       super(DeclFilter, self).__init__(condition_func = lambda node : type(node) == c_ast.Decl and (getattr(node, attribute) == value), prev_brother = prev_brother)



class FuncCallFilter(GenericFilterVisitor):
   """ Returns the first node with a FuncCall
   """

   def __init__(self, prev_brother = None):
       # The condition __doc__ is used as exception information
       def condition(node):
          """ FuncCall """
          return type(node) == c_ast.FuncCall
       super(FuncCallFilter, self).__init__(condition_func = condition , prev_brother = prev_brother)


class FuncDeclOfNameFilter(GenericFilterVisitor):
   """ Returns the first node with a FuncCall
   """

   def __init__(self, name, prev_brother = None):
       # The condition __doc__ is used as exception information
       def condition(node):
           """ FuncDecl """
           if type(node) == c_ast.FuncDecl:
#               print "Looking for : " + name.name
                pass
           return type(node) == c_ast.FuncDecl and getattr(node.parent, 'name') == name.name
       super(FuncDeclOfNameFilter, self).__init__(condition_func = condition , prev_brother = prev_brother)

