
import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


import config

def parse_source(template_source, template_name):
        """ Returns the AST from a string """
        p = subprocess.Popen("cpp -w -ansi -pedantic -CC -U __USE_GNU  -P -I " + config.INCLUDE_DIR + " -I " + config.FAKE_LIBC, shell=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        stripped_source = p.communicate(template_source)[0]
        ast = c_parser.CParser(lex_optimize = True, yacc_optimize = True).parse(stripped_source, filename = template_name)
	return ast

