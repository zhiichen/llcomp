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
	

import subprocess
from cStringIO import StringIO

from pycparser import c_parser, c_ast


print "Translating " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())
p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P -I /home/rreyes/llcomp/src/include/fake_libc_include/", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
clean_source = p.communicate(template_code)[0]
process = subprocess.Popen("sed -nf nocomments.sed", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
stripped_code = process.communicate(clean_source)[0]
ast = c_parser.CParser(lex_optimize = False, yacc_optimize = False).parse(stripped_code, filename = filename)

print " OK "


if not output_file:
	ast.show(attrnames = True)

# t = SimpleMutator()

new_ast = ast # t.apply(ast)


# Print the AST
v = CloneWriter(filename = output_file)
v.visit(new_ast)

del v

# Call pretty printer over the file
if output_file:
	import os
	if os.system("indent -kr " + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "


