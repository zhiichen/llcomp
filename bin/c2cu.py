from pycparser import parse_file

# from Visitors.clone_visitor import CUDAWriter

from Backends.CudaBackend.Writers.CUDAWriter import CUDAWriter

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



###################### First Layer  : File parsing

# Parse file
	
from Frontend.Parse import parse_source

print "Parsing " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())
ast = parse_source(template_code, filename)

print " OK "

print " Migrating to Internal Representation ...."


from Frontend.InternalRepr import AstToIR
# Transform the C ast into the internal representation
new_ast = AstToIR(Writer = CUDAWriter).transform(ast)


print " OK "

###################### Second Layer  : Transformation tools

print "Mutating ...",

# Optimize code
from Mutators.Optimizer import MatrixDeclToPtr, ConstantCalc


MatrixDeclToPtr(start_ast = new_ast).fast_apply_all(new_ast)

ConstantCalc().fast_apply_all(new_ast)

from Mutators.Cuda.Common import CudaMutatorError, CudaTransformer 

try:
   end_ast = CudaTransformer.apply(new_ast)
except CudaMutatorError as cme:
   print " Error while mutating tree "
   print cme

if end_ast:
	print " OK "
else:
	print " Translation interrupted due to previous errors "
	import sys
	sys.exit(-1)


print " OK "

## Print the AST

print " Writing result ...",

v = CUDAWriter(filename = output_file)
v.visit(end_ast)



print " OK "



# Call pretty printer over the file
if output_file:
	del v  # Close files opened by CUDA Writer
	import os
	if os.system("indent -kr " + config.WORKDIR + output_file) != 0:
		print " You need to install the indent tool to pretty print ouput files "
   


