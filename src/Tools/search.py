
from pycparser import c_ast
from Tools.Debug import DotDebugTool

from Tools.tree import NodeNotFound

from Visitors.generic_visitors import DeclFilter

def decl_of_id(id, ast):
        """ Returns the TypeDecl node of a given ID node """
        # TODO: Clean this code...
        act = id.parent
        decl = None
        while act != None:
                # Taking advantage of lazy boolean evaluation, if the first part is false, 
                #    the second is not executed, so, we won't have an exception
                if hasattr(act, 'decls') and act.decls:
                  for elem in act.decls:
                     try:
                        decl = DeclFilter(attribute = "name", value = id.name).apply(elem)
                     except NodeNotFound:
                        pass

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
        # Identifier not declared
        return None



def type_of_id(id, ast):
        """ Returns the TypeDecl node of a given ID node """
        return decl_of_id(id,ast).type


