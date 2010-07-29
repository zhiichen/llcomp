

from pycparser import c_parser, c_ast

import subprocess
from cStringIO import StringIO

from Backends.C.Writers.CWriter import CWriter

from Backends.Common.Visitors.GenericVisitors import IdentifierTypeFilter, TypedefFilter

import cStringIO


from Tools.Tree import NodeNotFound

from mako.template import Template as MakoTemplate



class TemplateParser(MakoTemplate):
    def __init__(self, *args, **kwargs):
        super(TemplateParser, self).__init__(*args, **kwargs)



class TemplateVarNode:

    def fast_write(self, elem):
        writeIO = cStringIO.StringIO()
        cw = CWriter(stream = writeIO)
        cw.visit(elem)
        return writeIO.getvalue()

    def decl_write(self, elem):
        tmp = " ".join(['%s'%stor for stor in elem.storage]) 
        if isinstance(elem.type, c_ast.ArrayDecl):
            tmp += " "
            tmp_node = elem.type
            while not (isinstance(tmp_node, c_ast.TypeDecl) or isinstance(tmp_node, c_ast.PtrDecl)) and tmp_node:
                tmp_node = tmp_node.type

            tmp += fast_write(tmp_node)
        else:
            tmp += " ".join(['%s'%qual for qual in elem.quals])
            tmp += " "
            tmp += self.fast_write(elem.type)
        return tmp


    def __init__(self, var, ast, name_func = lambda elem : elem.name, type_func = lambda elem: elem.type):
        self._var = var
        self._ast = ast
        self._type = var.type
        self._type_func = type_func
        self._name_func = name_func

    def __str__(self):
        return self._name_func(self._var)

    @property
    def name(self):
        return self._name_func(self._var)

    @property
    def type(self):
        return self._type_func(self._type)

    @property
    def declaration(self):
        return self.decl_write(self._var).replace(';','').replace('\n','')

    @property
    def numelems(self):
        numelems = "1"
        if isinstance(self.ptr.type, c_ast.ArrayDecl): 
            numelems = str(self.ptr.type.dim)
        return numelems

    @property
    def ptr(self):
        return self._var


def get_template_array(var_list, ast, func = lambda elem : True, name_func = lambda elem : elem.name, type_func = lambda elem : elem.type):
    """ Prepare template array for vars """
    names = []

    for elem in var_list:
        if func(elem):
            # Type string | var name | pointer to type | pointer to var | declaration string
            var = TemplateVarNode(var = elem, name_func = name_func, type_func = type_func, ast = ast)
            names.append(var)
    return names

# TODO: Clean (Move this to template class and generalize)
def get_typedefs_to_template(shared_vars,ast):
    """ Build a list with all type declarations for the shared var array """
    decls_dict = {}
    param_var_list = []
    for elem in shared_vars:
        try:
            identifier_type = IdentifierTypeFilter().apply(elem.type)
            if not identifier_type.names[0] in decls_dict:
                typedef_node = TypedefFilter(name = identifier_type.names[0]).apply(ast)
                # TODO: Avoid construction/destruction
                typedefIO = cStringIO.StringIO()
                cw = CWriter(stream = typedefIO)
                cw.visit(typedef_node)
                decls_dict[identifier_type.names[0]] = str(typedefIO.getvalue())
        except NodeNotFound as nnf:
            # It is not a complex type
#                print " Not a userdefined-type " + elem[1]
#                structIO = cStringIO.StringIO()
#                cw = CWriter(stream = structIO)
#                cw.visit(elem[3])
#                param_var_list.append(str(structIO.getvalue()).replace(';','') )
            pass

    return [elem for elem in decls_dict.values()]
