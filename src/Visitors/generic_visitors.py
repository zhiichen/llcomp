

class NodeNotFound(Exception):
   def __init__(self, node):
      self.node = node

   def __str__(self):
      return "Node " + str(self.node) + " not found "

class NodeNotValid(Exception):
   def __init__(self, node):
      self.node = node

   def __str__(self):
      return "Node " + str(self.node) + " not valid "

class PositionNotValid(Exception):
   def __init__(self):
      pass

   def __str__(self):
      return "Position not valid (choose either begin or end ) "







class FilterVisitor(object):
   """ Returns the first node matching the criteria 
   """

   def __init__(self, match_node_type, prev_brother = None):
       self.match_node_type = match_node_type;
       self.prev_brother = prev_brother;
       self.match = False
       self.parent_of_match = None

   def apply(self, ast):
      """ Apply filter to the ast """
      self.match = False
      node = ast
      if type(node) != self.match_node_type:
         node = self.generic_visit(ast)
      if type(node) != self.match_node_type:
         raise NodeNotFound(self.match_node_type)
      return node

   def visit(self, node, prev, offset = 1):
        """ Visit a node. 
        """
        if type(node) == self.match_node_type and (self.prev_brother != None and self.prev_brother == prev):
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
       iter = node.children().__iter__();
       r = node;
       prev = None;
       try:
          c = iter.next();
          while type(c) != self.match_node_type or (self.prev_brother != None and self.prev_brother != prev):
             r = self.visit(c, prev)
             if type(r) == self.match_node_type and (self.prev_brother != None and self.prev_brother == prev):
                # Stop iterating, we've found the mathing node
                # Do not execute the else code, we already have in r 
                # the matching node
                self.match = True
                break
             prev = c
             c = iter.next()
          else:
             # r is always the matching node (even if its in the same level)
             r = c
             self.match = True
       except StopIteration:
          pass

       if (self.match == True and self.parent_of_match == None):
           self.parent_of_match = node

       return r

   def parentOfMatch(self):
       return self.parent_of_match





class AttributeFilter(object):
   """ Returns the first node with a given attribute
   """

   def __init__(self, match_attribute, prev_brother = None):
       self.match_attribute = match_attribute;
       self.prev_brother = prev_brother;
       self.match = False
       self.parent_of_match = None

   def apply(self, ast):
      """ Apply filter to the ast """
      self.match = False
      node = ast
      if not hasattr(node, self.match_attribute):
         node = self.generic_visit(ast)
      if not hasattr(node, self.match_attribute):
         raise NodeNotFound(self.match_attribute)
      return node

   def visit(self, node, prev, offset = 1):
        """ Visit a node. 
        """
        if hasattr(node, self.match_attribute) and (self.prev_brother != None and self.prev_brother == prev):
           self.match = True
           return node
        # Continue the search....
        method = 'visit_' + node.__class__.__name__
        print " Visiting .... " + node.__class__.__name__ + " Method : " + method
        print " Node : " + str(node)
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, offset)
        
   def generic_visit(self, node, offset = 0):
       """ Called if no explicit visitor function exists for a 
           node. Implements preorder visiting of the node.
       """
       print " Generic visit of " + str(node)
       # Store the parent node of the match
       iter = None
       if type(node) == type([]):
          print "Type : " + str(type(node)) + " == " + str(type([]))
          print " Node attr : " + str(dir(node))
          iter = node.__iter__();
       else:
          print "Type : Node "
          print " Node attr : " + str(dir(node))
          iter = node.children().__iter__();
       r = node;
       prev = None;
       try:
          c = iter.next();
          print  " Check " + str(c)  + " Attr " + str(dir(c))
          while not hasattr(c, self.match_attribute) or (self.prev_brother != None and self.prev_brother != prev):
             print  " Check " + str(c)
             r = self.visit(c, prev)
             if hasattr(r, self.match_attribute) and (self.prev_brother != None and self.prev_brother == prev):
                # Stop iterating, we've found the mathing node
                # Do not execute the else code, we already have in r 
                # the matching node
                self.match = True
                break
             prev = c
             c = iter.next()
          else:
             # r is always the matching node (even if its in the same level)
             r = c
             self.match = True
       except StopIteration:
          pass

       if (self.match == True and self.parent_of_match == None):
           self.parent_of_match = node
       print "r : " + str(r)

       return r

   def parentOfMatch(self):
       return self.parent_of_match




