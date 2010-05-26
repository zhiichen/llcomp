

from pycparser import c_ast

class NodeNotFound(Exception):
   def __init__(self, node):
      self.node = node

   def __str__(self):
      return "Node " + str(self.node) + " not found "

class NodeNotValid(Exception):
   def __init__(self, node):
      self.node = node

   def __str__(self):
      return "Node " + str(self.node) + " not valid (type: " + str(type(self.node)) + ")"

class PositionNotValid(Exception):
   def __init__(self):
      pass

   def __str__(self):
      return "Position not valid (choose either begin or end ) "





# InsertVisitor(mark_node = parent_stmt, subtree = maxThreadNumber_subtree, position = "mark", method = "append").apply(ast)
class InsertTool:
    """ Inserts the childs of a node (a subtree) on the given position. It doesn't insert the parent node of the new subtree. """
    def __init__(self, subtree = None, position = "begin", node = None):
       """ Inserter visitor """
#       self.mark_node = mark_node
       self.subtree = subtree
       self.position = position # Options: begin, end
       self.node = node
       self.place = 0

    def apply(self, target_node, attribute_name):
       """ Insert the object subtree inside the attribute node of the target_node """
       attr = getattr(target_node, attribute_name)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_node)
       # Find the insert place
#       from Tools.Debug import DotDebugTool
#       DotDebugTool(select_node = self.subtree).apply(target_node)

       if self.node:
           self.place = attr.index(self.node)
       # Insert the node
       if self.position == "begin":
           # Funny trick to insert as first element: first reverse, then insert on 0
           childrens = list(self.subtree.children())
           childrens.reverse()
           for it in childrens:
               attr.insert(self.place, it)
       elif self.position == "end":
           attr.extend(self.subtree.children())
       else:
           raise PositionNotValid
       setattr(target_node, attribute_name, attr)
       return target_node


# ReplaceTool(subtree = kernelLaunch_subtree, replaced_node = parent_stmt).apply(parent_stmt, 'stmts')
class ReplaceTool:
    """ Replace a subtree with another """
    def __init__(self, new_node, old_node):
       """ Replace visitor """
       self.new_node = new_node
       self.old_node = old_node 

    def apply(self, target_node, attribute_name):
       """ Replace self.old_node with self.new_node """
       attr = getattr(target_node, attribute_name)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_node)
#       print "************************** "
#       for elem in attr:
#         print str(type(elem)) + " --> ",
#         if type(elem) == type(""):
#            print elem
#         elif hasattr(elem, 'name') and type(elem.name) == type(""):
#            print elem.name
#         elif hasattr(elem, 'name') and hasattr(elem.name, 'name') and type(elem.name.name) == type(""):
#            print elem.name.name
#         elif type(elem) == c_ast.FuncDef:
#            if type(elem.decl.name) == type(""):
#               print elem.decl.name
#            else:
#               print elem.decl.name.name
#         else:
#            print elem
#       print "Old_node : " + str(self.old_node)
#       print "************************** "
       position = attr.index(self.old_node)
       attr[position] = self.new_node
       setattr(target_node, attribute_name, attr)
       return target_node

class RemoveTool:
    """ Remove a subtree """
    def __init__(self, target_node):
       """ Remove visitor """
       self.target_node = target_node

    def apply(self, target_subtree, attribute_name):
       """ Remove a subtree """
       if self.target_node == None: return target_subtree
       attr = getattr(target_subtree, attribute_name)
       position = attr.index(self.target_node)
       # 1. Check attribute is a list of nodes
       if not type(attr) == type([]):
           raise NodeNotValid(target_subtree)
       del attr[position] 
       setattr(target_subtree, attribute_name, attr)
       return target_subtree





# Parent link:

def link_all_parents(ast):
        """ Function to link the nodes of the AST in reverse order, using a parent attribute in each node """
        def deep_first_search(root, visited = None,
                preorder_process  = lambda x: None):
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
        def link_parent(node):
                for child in node.children():
                        child.parent = node

        deep_first_search(root = ast, visited = None, preorder_process = link_parent)





     
