
import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


def parse_template(template_source, template_name):
        """ Returns the AST from a string """
        p = subprocess.Popen("cpp -w -ansi -pedantic -CC -U __USE_GNU  -P -I /home/rreyes/llcomp/src/include/ -I /home/rreyes/llcomp/src/include/fake_libc_include/", shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        stripped_source = p.communicate(template_source)[0]
        ast = c_parser.CParser(lex_optimize = False, yacc_optimize = False).parse(stripped_source, filename = template_name)
	return ast

