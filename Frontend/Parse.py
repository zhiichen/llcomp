# -*- encoding: utf8 -*-
"""

.. module:: Parse
    :synopsis: Parsing utilities

.. moduleauthor:: Ruymán Reyes Castro <rreyes@ull.es>


"""


import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast
import config

def parse_source(code_source, code_name):
    """Parse the  source string using cpp as preprocessor and pycparser

        :param code_source: string with the source code
        :param code_name: string with the code name
        :rtype: ast FileAST of the source code

        .. warning::
            This function requires *cpp* to be in PATH.

    """
    # Prepare preprocessor pipe
    p = subprocess.Popen("cpp -w -ansi -pedantic -CC -U __USE_GNU  -DLLC_TRANSLATION -P -I " + 
        config.INCLUDE_DIR + " -I " + config.FAKE_LIBC, shell=True, bufsize=1, 
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    # Send / Retrieve string to the preprocessor
    stripped_source = p.communicate(code_source)[0]
    # Parse the file and get the AST
    ast = c_parser.CParser(lex_optimize = True, yacc_optimize = True).parse(
        stripped_source, filename = code_name)
    return ast

