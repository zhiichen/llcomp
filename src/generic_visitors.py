

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
      node = self.generic_visit(ast)
      if type(node) != self.match_node_type:
         raise NodeNotFound(self.match_node_type)
      return node

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
           print " Parent: ";
           node.show()

       return r

   def parentOfMatch(self):
       return self.parent_of_match


# InsertVisitor(mark_node = parent_stmt, subtree = maxThreadNumber_subtree, position = "mark", method = "append").apply(ast)
class InsertTool:
    """ Inserts a subtree on the given order """
    def __init__(self, subtree = None, position = "begin"):
       """ Inserter visitor """
#       self.mark_node = mark_node
       self.subtree = subtree
       self.position = position # Options: begin, end

    def apply(self, target_node, attribute_name):
       """ Insert the object subtree inside the attribute node of the target_node """
       attr = getattr(target_node, attribute_name)
       print " *** Target node *** " 
       target_node.show()
       print " ********************** "
       print " Node to be grafted "
       self.subtree.show()
       print dir(self.subtree)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_node)
       if self.position == "begin":
           # Funny trick to insert to the begin, first reverse, then insert on 0
           childrens = list(self.subtree.children())
           childrens.reverse()
           for it in childrens:
               attr.insert(0, it)
       elif self.position == "end":
           attr.extend(self.subtree.children())
       else:
           raise PositionNotValid
       print str(attr)
       setattr(target_node, attribute_name, attr)
       print " *** New subtree *** " 
       target_node.show()
       print " ********************** "
       return target_node


# ReplaceVisitor(subtree = kernelLaunch_subtree, relpaced_node = parent_stmt).apply(parent_stmt, 'stmts')
class ReplaceTool:
    """ Replace a subtree with another """
    def __init__(self, new_node, old_node):
       """ Replace visitor """
       self.new_node = new_node
       self.old_node = old_node 

    def apply(self, target_node, attribute_name):
       """ Replace self.old_node with self.new_node """
       attr = getattr(target_node, attribute_name)
       print " ********************** "
       position = attr.index(self.old_node)
       print " Element position: " + str(position)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_node)
       attr[position] = self.new_node
       setattr(target_node, attribute_name, attr)
       return target_node

class RemoveTool:
    """ Replace a subtree with another """
    def __init__(self, target_node):
       """ Replace visitor """
       self.target_node = target_node

    def apply(self, target_subtree, attribute_name):
       """ Replace self.old_node with self.new_node """
       attr = getattr(target_subtree, attribute_name)
       position = attr.index(self.target_node)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_subtree)
       del attr[position] 
       setattr(target_subtree, attribute_name, attr)
       return target_subtree









     
