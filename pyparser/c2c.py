from pycparser import parse_file

from clone_visitor import CloneVisitor

from SimpleMutator import SimpleMutator

from sys import argv, exit


if len(argv) > 1:
	filename  = argv[1]
else:
	print ">>> File not found!"
	exit()

# Parse file
	
ast = parse_file(filename, use_cpp=True, cpp_path='/usr/bin/cpp',
	cpp_args=r'-I../utils/fake_libc_include');

ast.show(attrnames = True)

# t = SimpleMutator()

new_ast = ast # t.apply(ast)


# Print the AST
v = CloneVisitor()
v.visit(new_ast)
