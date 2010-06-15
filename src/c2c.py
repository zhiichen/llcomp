from pycparser import parse_file

from Visitors.clone_visitor import CWriter, OmpWriter

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

from Tools.Parse import parse_template
print "Translating " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())
ast = parse_template(template_code, filename)



print " OK "

if not output_file:
	ast.show(attrnames = True)

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



link_all_parents(ast)
# t = SimpleMutator()

new_ast = ast # t.apply(ast)

# Optimize code
from Mutators.Optimizer import MatrixDeclToPtr, ConstantCalc

MatrixDeclToPtr(start_ast = new_ast).fast_apply_all(new_ast)

ConstantCalc().fast_apply_all(new_ast)

# Print the AST
v = OmpWriter(filename = output_file)
v.visit(new_ast)

del v

# Call pretty printer over the file
if output_file:
	import os
	if os.system("indent -kr " + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "


