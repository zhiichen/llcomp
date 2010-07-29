from pycparser import c_ast

import sys


from Backends.Common.Writers.CommonWriter import OffsetNodeVisitor

from Tools.Declarations import type_of_node

class FlopCountVisitor(OffsetNodeVisitor):
    """ Generates FlopCount code from the IR

    """
    inside = False

    def __init__(self, ast = None):
        self._ast = ast

    # ********** Visit **********
    def visit_BinaryOp(self, node, offset = 0):
        print "*** " + str(node) + " ***"
        print " Type of node :"  + str(type_of_node(node, self._ast))
        print "Dir : " + str(dir(node)) + " ***"       
        node.show()

#
#
#        if (isinstance(node.left, c_ast.BinaryOp)):
#            self.write(offset, "(")
#            self.visit(node.left)
#            self.write(offset, ")")
#        else:
#            self.visit(node.left)
#
#        self.write_blank()
#        self.write(0, node.op)
#        self.write_blank()
#        if (isinstance(node.right, c_ast.BinaryOp)):
#            self.write(offset, "(")
#            self.visit(node.right)
#            self.write(offset, ")")
#        else:
#            self.visit(node.right)
#        # self.write_blank()
#
#                
#


