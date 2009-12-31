

class NodeNotFound(Exception):
   def __init__(self, node):
      self.node = node

   def __str__(self):
      return "Node " + type(self.node) + " not found "

class FilterVisitor(object):
   """ Returns the first node matching the criteria 
   """

   def __init__(self, match_node_type, prev_brother = None):
       self.match_node_type = match_node_type;
       self.prev_brother = prev_brother;
       self.match = False

   def visit(self, node, prev, offset = 1):
        """ Visit a node. 
        """
 #       print " Visiting: " + node.__class__.__name__
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
       iter = node.children().__iter__();
       r = node;
       prev = None;
       try:
          c = iter.next();
          while type(c) != self.match_node_type:
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
       if not self.match:
          raise NodeNotFound(self.match_node_type)
       return r


