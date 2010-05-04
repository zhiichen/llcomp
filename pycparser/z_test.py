from pycparser import c_parser, c_ast

source_code = """

int x;
#pragma omp threadprivate(x)

int main() {
    int i, l;

    #pragma omp parallel private(i) shared(l, a) 
    {
      for  (i = 0/*Pututu*/; i <= 10; i++) {
         a = 0;
      }
// tetete
	    #pragma omp for reduction (+ : a)
	    for (i = 0; i <= 10; i++) {
		a = a + i;
	    }
    }
}

"""

import subprocess
from cStringIO import StringIO


try:
    p = subprocess.Popen("cpp -ansi -pedantic -CC -U __USE_GNU  -P", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
except IOError:
    exit(0)
clean_source = p.communicate(source_code)[0]

#try:
#	process = subprocess.Popen("sed -nf /home/rreyes/pycparser-read-only/nocomments", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#except IOError:
#    exit(0)
# stripped_code = process.communicate(clean_source)[0]
stripped_code = clean_source


parser = c_parser.CParser(
            lex_optimize=False,
            yacc_optimize=False,
            yacc_debug=True,
)
ast = parser.parse(stripped_code, filename = 'tutu')
ast.show()

# function_body = ast.ext[-1].body #hardcoded to the main() function

print ast.ext[-1].body.stmts[0].show()

#for stmt in function_body.stmts:
#    print stmt.coord, stmt
    
    
#~ class StructRefVisitor(c_ast.NodeVisitor):
    #~ def visit_StructRef(self, node):
        #~ print node.name.name, node.field.name


#~ parser = c_parser.CParser()
#~ ast = parser.parse(code)

#~ ast.show()

#~ v = StructRefVisitor()
#~ v.visit(ast)

