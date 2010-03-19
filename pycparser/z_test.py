from pycparser import c_parser, c_ast

source_code = """

int a[30];

int main() {
    int __b__;

    #pragma omp parallel for shared(a)
    for (i = 0; i <= 10; i++) {
	a = a + i;
	a;
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

try:
	process = subprocess.Popen("sed -nf /home/ruyk/pycparser-read-only/nocomments", shell = True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
except IOError:
    exit(0)
stripped_code = process.communicate(clean_source)[0]


parser = c_parser.CParser(
            lex_optimize=False,
            yacc_optimize=False,
            yacc_debug=True,
)
ast = parser.parse(stripped_code, filename = 'tutu')
# ast.show()

# function_body = ast.ext[-1].body #hardcoded to the main() function

# print ast.ext[-1].body.stmts[0].show()

#for stmt in function_body.stmts:
#    print stmt.coord, stmt


def link_all_parents(ast):
	""" Function to link the nodes of the AST in reverse order, using a parent attribute in each node """
	def deep_first_search(root, visited = None, 
		preorder_process  = lambda x: None):
		"""
		Given a starting vertex, root, do a depth-first search.
		"""
		to_visit = []  # a list can be used as a stack in Python
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

# ast.ext[-1].body.decls[0].show()

def type_of_id(id, ast):
	""" Returns the TypeDecl node of a given ID node """
	act = id.parent
	while act != None:
		# Taking advantage of lazy boolean evaluation, if the first part is false, 
		#    the second is not executed, so, we won't have an exception
		if hasattr(act, 'decls') and act.decls:
			# look on the decls section 
			for decl in act.decls:
				if decl.name == id.name:
					return decl.type
		if hasattr(act, 'ext') and act.ext:
			for decl in act.ext:
				if decl.name == id.name:
					return decl.type
		# Keep going up
#		print str(type(act)) + "==" + str(dir(act)) + " ==> " + str(type(act.parent))
		act = act.parent
	# Identifier not declared
	return None

typedecl = type_of_id(ast.ext[-1].body.stmts[-1].stmt.stmts[-1], ast)


def param_for_type(typedecl):
	""" Returns the parameter expression for a given type declaration """
	typedecl.show()


param_for_type(typedecl)
# ast.show()

