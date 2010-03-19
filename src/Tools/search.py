
def decl_of_id(id, ast):
        """ Returns the TypeDecl node of a given ID node """
        act = id.parent
        while act != None:
                # Taking advantage of lazy boolean evaluation, if the first part is false, 
                #    the second is not executed, so, we won't have an exception
                if hasattr(act, 'decls') and act.decls:
                        # look on the decls section 
                        for decl in act.decls:
                                if decl.name == id.name:
                                        return decl
                if hasattr(act, 'ext') and act.ext:
                        for decl in act.ext:
                                if decl.name == id.name:
                                        return decl
                # Keep going up
#               print str(type(act)) + "==" + str(dir(act)) + " ==> " + str(type(act.parent))
                act = act.parent
        # Identifier not declared
        return None



def type_of_id(id, ast):
        """ Returns the TypeDecl node of a given ID node """
        return decl_of_id(id,ast).type


