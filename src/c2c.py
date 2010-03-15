from pycparser import parse_file

from Visitors.clone_visitor import CloneWriter

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
	
ast = parse_file(filename, use_cpp=False, cpp_path='/usr/bin/cpp',
	cpp_args=r'-ansi -fdirectives-only -I fake_libc_include/'); #, lex_optimize=False, yacc_optimize=False, yacc_debug=True);

if not output_file:
	ast.show(attrnames = True)

# t = SimpleMutator()

new_ast = ast # t.apply(ast)


# Print the AST
v = CloneVisitor(filename = output_file)
v.visit(new_ast)

del v

# Call pretty printer over the file
if output_file:
	import os
	if os.system("indent -kr " + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "


