

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
       setattr(target_node, attribute_name, attr)
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
       position = attr.index(self.old_node)
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









     
