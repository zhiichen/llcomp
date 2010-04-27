from pycparser import parse_file

from Visitors.clone_visitor import CUDAWriter

from Mutators.Cuda import CM_OmpParallel, CudaMutator


from sys import argv, exit

import config

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


print " Backward link ....",

link_all_parents(ast)


print " OK "

#if not output_file:
#	ast.show(attrnames = True)

print "Mutating ..."

# Optimize code
from Mutators.Optimizer import ConstantCalc

ConstantCalc().fast_apply_all(ast)


t = CM_OmpParallel()
# t = CudaMutator()

new_ast = t.apply(ast)

if new_ast:
	print " OK "
else:
	print " ERROR "
	import sys
	sys.exit(-1)

print " Update backward link ....",

link_all_parents(ast)


print " OK "

## Print the AST

print " Writing result ...",

v = CUDAWriter(filename = output_file)
v.visit(new_ast)



print " OK "



# Call pretty printer over the file
if output_file:
	del v  # Close files opened by CUDA Writer
	import os
	if os.system("indent -kr " + config.WORKDIR + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "
   


