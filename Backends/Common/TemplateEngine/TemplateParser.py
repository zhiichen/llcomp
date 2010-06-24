

from pycparser import c_parser, c_ast

import subprocess
from cStringIO import StringIO

from Backends.CBackend.Writers.CWriter import CWriter

import cStringIO

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
