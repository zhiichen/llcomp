

class FilterVisitor(object):
   """ Returns the first node matching the criteria 
   """

   def __init__(self, match_node, prev_brother):
       self.match_node = match_node;
       self.prev_brother = prev_brother;

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
       prev = None;
       try:
          prev = c
          c = iter.next();
          while type(c) != self.match_node:
             r = self.visit(c)
             if type(r) == self.match_node:
                # Stop iterating, we've found the mathing node
                # Do not execute the else code, we already have in r 
                # the matching node
                break
             prev = c
             c = iter.next()
          else:
             # r is always the matching node (even if its in the same level)
             r = c
       except StopIteration:
          pass
       return r


