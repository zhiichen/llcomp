from pycparser import parse_file

from Visitors.clone_visitor import CUDAWriter

from Mutators.Cuda import CudaMutator


from sys import argv, exit

output_file = None

if len(argv) > 1:
	filename  = argv[1]
	if len(argv) > 2 :
		output_file = argv[2]

else:
	print ">>> File not found!"
	exit()

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



# Parse file
	
ast = parse_file(filename, use_cpp=True, cpp_path='/usr/bin/cpp',
	cpp_args=r'-ansi -I../utils/fake_libc_include'); #, lex_optimize=False, yacc_optimize=False, yacc_debug=True);


link_all_parents(ast)

#if not output_file:
#	ast.show(attrnames = True)

t = CudaMutator()

new_ast = t.apply(ast)


link_all_parents(ast)

## Print the AST
v = CUDAWriter(filename = output_file)
v.visit(new_ast)



# del v

# Call pretty printer over the file
if output_file:
	import os
	if os.system("indent -kr " + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "


