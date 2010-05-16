from pycparser import parse_file

from Visitors.clone_visitor import CUDAWriter

from Mutators.Cuda import CudaMutator, CudaMutatorError
from Mutators.CM_OmpFor import CM_OmpFor
from Mutators.CM_OmpParallel import CM_OmpParallel

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

from pycparser import c_ast

from Tools.Parse import parse_template

print "Translating " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())
ast = parse_template(template_code, filename)

print " OK "


print " Backward link ....",

link_all_parents(ast)


print " OK "

#if not output_file:
#	ast.show(attrnames = True)

print "Mutating ..."

# Optimize code
from Mutators.Optimizer import MatrixDeclToPtr, ConstantCalc


MatrixDeclToPtr(start_ast = ast).fast_apply_all(ast)

ConstantCalc().fast_apply_all(ast)

new_ast = None

try:
   # t = CudaMutator()
   t = CM_OmpParallel(kernel_prefix='compute')
   new_ast = t.apply(ast)
   link_all_parents(new_ast)
   t2 = CM_OmpParallel(kernel_prefix='update')
   new_ast = t2.apply(new_ast)
except CudaMutatorError as cme:
   print " Error while mutating tree "
   print cme

if new_ast:
	print " OK "
else:
	print " Translation interrupted due to previous errors "
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
   


