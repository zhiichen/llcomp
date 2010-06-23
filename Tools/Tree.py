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
        return "Position not valid (choose either begin or end)"



class InsertTool:
     """Inserts the childs of a node (a subtree) on the given position. 

         .. note:: 
                It doesn't insert the parent node of the new subtree. 


         Example:
            >>> InsertTool(subtree = kernel_init_subtree, position = "begin").apply(cuda_stmts, 'decls')

     """
     def __init__(self, subtree = None, position = "begin", node = None):
         """Prepare insertion """
         self.subtree = subtree
         self.position = position # Options: begin, end
         self.node = node
         self.place = 0

     def apply(self, target_node, attribute_name):
         """ Apply the insertion on the attribute of the target node
    
                :param target_node: Node where insert the subtree
                :param attribute_name: Attribute of the target_node where the subtree is to be inserted

                :return: Target node after insertion
         """
         attr = getattr(target_node, attribute_name)
         # Check attribute is a list of nodes
         if not type(attr) == type([]):
              raise NodeNotValid(target_node)
         # Find the place to insert
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
         # Update the target_node with the new_subtree
         setattr(target_node, attribute_name, attr)
         return target_node


class ReplaceTool:
     """ Replace a subtree with another 


         >>> ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')

     """
     def __init__(self, new_node, old_node):
         """ Prepare the replace """
         self.new_node = new_node
         self.old_node = old_node 

     def apply(self, target_node, attribute_name):
         """ Apply the replace, effectively changing self.old_node by self.new_node 

              :param target_node: parent node of old_node
              :param attribute_name: Attribute name which contains the old_node on the parent

              :return: Target Node after replace
         """
         attr = getattr(target_node, attribute_name)
         # 1. Check attribute is a list of nodes
         if not type(attr) == type([]):
              raise NodeNotValid(target_node)

         position = attr.index(self.old_node)
         attr[position] = self.new_node
         setattr(target_node, attribute_name, attr)
         return target_node

class RemoveTool:
     """ Remove a subtree from the AST

     """
     def __init__(self, target_node):
         """ Set the target node """
         self.target_node = target_node

     def apply(self, target_subtree, attribute_name):
         """ Apply the removal

                :param target_subtree: target_subtree where target_node resides
                :param attribute_name: Name of the attribute where the target_node resides in the target_subtree
                :return: Target subtree (container of the removed target_node)
         """
         if self.target_node == None: return target_subtree
         attr = getattr(target_subtree, attribute_name)
         position = attr.index(self.target_node)
         # 1. Check attribute is a list of nodes
         if not type(attr) == type([]):
              raise NodeNotValid(target_subtree)
         del attr[position] 
         setattr(target_subtree, attribute_name, attr)
         return target_subtree


