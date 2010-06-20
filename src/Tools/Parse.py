# -*- encoding: utf8 -*-

"""
Parse - Module with parsing utilities


@author Ruymán Reyes Castro

"""


import subprocess
from cStringIO import StringIO
from pycparser import c_parser, c_ast


import config

def parse_source(template_source, template_name):
   """ Parse the  source string using cpp as preprocessor and pycparser

      @return ast FileAST of the source code
   """
   # Prepare preprocessor pipe
   p = subprocess.Popen("cpp -w -ansi -pedantic -CC -U __USE_GNU  -P -I " + 
      config.INCLUDE_DIR + " -I " + config.FAKE_LIBC, shell=True, bufsize=1, 
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
   # Send / Retrieve string to the preprocessor
   stripped_source = p.communicate(template_source)[0]
   # Parse the file and get the AST
   ast = c_parser.CParser(lex_optimize = True, yacc_optimize = True).parse(
      stripped_source, filename = template_name)
   return ast

