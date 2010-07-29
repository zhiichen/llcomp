
from pycparser import c_ast
from Tools.Debug import DotDebugTool

from Tools.Tree import NodeNotFound

from Backends.Common.Visitors.GenericVisitors import DeclFilter

def decl_of_id(id, ast):
          """Returns the TypeDecl node of a given ID node 
          
                Because we don't have a symbol table, we transverse the tree using
                the parent link attribute, in order to find the declaration of the identifier.
                This process is slow, a symbol table needs to be implemented.

                :param id: ID Node
                :param ast: Pointer to start reverse search

                :return: Decl node of the identifier
          """
          act = id.parent
          decl = None
          
          while act != None:
                     # Check for a decl of the var as parameter
                     if isinstance(act, c_ast.FuncDef):
                        if act.decl.type.args and act.decl.type.args.params:
                            for elem in act.decl.type.args.params:
                                try:
                                    decl = DeclFilter(attribute = "name", value = id.name).apply(elem)
                                except NodeNotFound:
                                    pass
                     # Taking advantage of lazy boolean evaluation, if the first part is false, 
                     #     the second is not executed, so, we won't have an exception
                     if hasattr(act, 'decls') and act.decls:
                        for elem in act.decls:
                            try:
                                decl = DeclFilter(attribute = "name", value = id.name).apply(elem)
                            except NodeNotFound:
                                pass
                     # The uppermost level is the FileAST, where decls are on the ext attribute
                     if hasattr(act, 'ext') and act.ext:
                        for elem in act.ext:
                            try:
                                decl = DeclFilter(attribute = "name", value = id.name).apply(elem)
                            except NodeNotFound:
                                pass
     
                     if type(decl) == c_ast.Decl:
                        return decl

                     # Keep going up
                     act = act.parent

          # TODO: Raise identifier not declared
          return None



def type_of_id(id, ast):
          """Returns the TypeDecl node of a given ID node 

                Calls decl_of_id but returns the type argument

                :rtype: TypeDecl node of a given ID node 

          """
          return decl_of_id(id,ast).type



from Backends.Common.Writers.CommonWriter import OffsetNodeVisitor

class NodeTypeVisitor(OffsetNodeVisitor):
    # Priority among C types
    _type_priority = ['long double', 'double','float', 'long int', 'int', 'short int']


    def __init__(self, ast = None):
        self._ast = ast

    def get_type_name(self, elem):
        """ Return a list of names for a type """
        while not hasattr(type, 'type') and not hasattr(type, 'name'):
            type = type.type

        type = type_of_id(elem, self._ast)
        while not hasattr(type, 'names') and not hasattr(type, 'name'):
            type = type.type
        if hasattr(type, 'names'):
            return ' '.join(type.names)
        else:
            return str(type.name)


    def coherce_type(self, typeA, typeB):
        if self._type_priority.index(self.get_type_name(typeA)) > self._type_priority.index(self.get_type_name(typeB)):
            return typeA
        else:
            return typeB

    def visit_BinaryOp(self, node, offset = 0):
        typeA = self.visit(node.left)
        typeB = self.visit(node.right)
        return self.coherce_type(typeA, typeB)



def type_of_node(node, ast):
          """Returns the TypeDecl node of an expression


                Rules are: double>float>int

                :rtype: TypeDecl node of a given expression node

          """
          return NodeTypeVisitor(ast = ast).visit(node)

