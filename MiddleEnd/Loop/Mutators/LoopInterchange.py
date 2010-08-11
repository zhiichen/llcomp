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
    def __init__(self, *args, **kwargs):
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
        """ Mutator code 

            :param ast: node returned by the filter
        """
        interchange_parent = ast.parent.parent  # Note that the parent of any llc node is a pragma node
        interchange_node = ast
        first_loop = ast.loop
        second_loop = ast.loop.stmt
        
        # Second loop is now the first
        # TODO: Replace should support unknown parent attributes
        ReplaceTool(new_node = second_loop, old_node = interchange_node.parent).apply(interchange_parent, 'stmts')
        # TODO: Replace should update parent links?
        second_loop.parent = interchange_parent
        # Move the contents of the loop to the first loop
        first_loop.stmt = second_loop.stmt
        first_loop.parent = second_loop
        # Change the new outer loop statements to the new inner loop
        second_loop.stmt = first_loop
        del interchange_node
        # Return the new outer loop
        return second_loop


