from pycparser import parse_file

from Visitors.dot_visitor import DotWriter

from Mutators.Simple import SimpleMutator

from sys import argv, exit

output_file = None

if len(argv) > 1:
	filename  = argv[1]
	if len(argv) > 2 :
		output_file = argv[2]

else:
	print ">>> File not found!"
	exit()




# Parse file
	

import subprocess
from cStringIO import StringIO

from pycparser import c_parser, c_ast


# Parent link:

def link_all_parents(ast):
        """ Function to link the nodes of the AST in reverse order, using a parent attribute in each node """
        def deep_first_search(root, visited = None,
                preorder_process  = lambda x: None):
                """
                Given a starting vertex, root, do a depth-first search.
                """
                to_visit = [] 
                if visited is None: visited = set()

                to_visit.append(root) # Start with root
                while len(to_visit) != 0:
                        v = to_visit.pop()
                        if v not in visited:
                                visited.add(v)
                                preorder_process(v)
                                to_visit.extend(v.children())
        def link_parent(node):
                for child in node.children():
                        child.parent = node

        deep_first_search(root = ast, visited = None, preorder_process = link_parent)


print "Translating " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())

from Tools.Parse import parse_template
ast = parse_template(template_code, filename)



print " OK "


#if not output_file:
#	ast.show(attrnames = True)


link_all_parents(ast)

new_ast = ast 

#from Visitors.generic_visitors import FuncCallFilter_Iterable
#for elem in FuncCallFilter_Iterable().iterate(new_ast):
#   elem.show()


# from Visitors.generic_visitors import FuncCallFilter


# print " New  "
# for elem in FuncCallFilter().dfs_iter(ast):
#   print "Elem: " + str(elem.name.name)

from Mutators.Optimizer import ConstantCalc

ConstantCalc().fast_apply_all(ast)

#ConstantBinaryExpressionFilter().apply(ast)

# Print the AST
v = DotWriter(filename = output_file, highlight = [ast])
v.visit(new_ast)

# Ensure file is closed and saved (calling destructor)
del v


