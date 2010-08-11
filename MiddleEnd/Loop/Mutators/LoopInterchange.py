from pycparser import c_parser, c_ast


from Backends.Common.Visitors.GenericVisitors import * 

from Backends.Common.Mutators.AbstractMutator import *

from Tools.Tree import InsertTool, RemoveTool, ReplaceTool


class LoopInterchangeFilter(GenericFilterVisitor):
    """ Returns the first node with the given attribute
    """
    def __init__(self):
        def condition(node):
            if type(node) == c_ast.llcInterchange:
                return True
            return False
        super(LoopInterchangeFilter, self).__init__(condition_func = condition)

class LoopInterchange(AbstractMutator):
    """ Apply Loop Interchange """ 
    def __init__(self):
        super(LoopInterchange, self).__init__()

    def filter(self, ast):
        """  """
        raise NotImplemented

    def filter_iterator(self, ast):
        """ Fast filter  """
        return NotImplemented

    def fast_filter(self, ast):
        """ Fast filter , looking for binary expressions """
        return LoopInterchangeFilter().dfs_iter(ast)

    def mutatorFunction(self, ast):
        """ Mutator code """
        return ast



