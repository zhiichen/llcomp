
from pycparser import c_ast
from Tools.Debug import DotDebugTool

from Tools.Tree import NodeNotFound

from Visitors.generic_visitors import DeclFilter

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
                #    the second is not executed, so, we won't have an exception
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


