from pycparser import parse_file

from Visitors.clone_visitor import CWriter, OmpWriter

from sys import argv, exit

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
from Tools.Parse import parse_source
print "Parsing " + filename + " .... ", 
template_code = " ".join(open(filename, 'r').readlines())
ast = parse_source(template_code, filename)



print " OK "



print " Migrating to Internal Representation ....", 


from Frontend.InternalRepr import AstToIR

# Transform the C ast into the internal representation
new_ast = AstToIR(writer = OmpWriter).transform(ast)


print " OK "

###################### Second Layer  : Transformation tools


# Optimize code
from Mutators.Optimizer import MatrixDeclToPtr, ConstantCalc

MatrixDeclToPtr(start_ast = new_ast).fast_apply_all(new_ast)

ConstantCalc().fast_apply_all(new_ast)

############################################3
# Write file

# Call pretty printer over the file
if output_file:
   v = OmpWriter(filename = output_file)
   v.visit(new_ast)
   del v  # Ensure file closing
   import os
   if os.system("indent -kr " + output_file) != 0:
      print " You need to install the indent tool to pretty print ouput files "
else:
   new_ast.show(attrnames = True)
   print "************"
   print ast

