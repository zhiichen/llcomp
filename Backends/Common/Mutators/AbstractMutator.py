from pycparser import c_parser, c_ast
from Backends.Common.Visitors.GenericVisitors import *

from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.Declarations import type_of_id, decl_of_id

from Tools.Debug import DotDebugTool
from Frontend.Parse import parse_source

from Backends.Common.TemplateEngine.TemplateParser import TemplateParser, get_template_array

from pycparser import  c_ast





class MutatorException(Exception):
   """ Generic mutator exception """
   def __init__(self, description):
      self.description = description 
  
   def __str__(self):
      return self.description

class IgnoreMutationException(MutatorException):
   """ Exception raised when, for some reason, we need to stop a mutation but it is not an error 
      
   """
   pass

class AbortMutationException(MutatorException):
   """ Abort mutation, error """
   pass

class AbstractMutator(object):
   """ Abstract class representing a mutation.


   """
   def __init__(self):
      pass

   def filter(self, ast):
      """ Calls to a simple filter """
      pass

   def mutatorFunction(self, ast):
      """ Mutates the AST

          @return Starting point of the mutation
      """
      return ast

   def apply(self, ast, mutator_opt_arg = None):
      """ Apply a mutation 
   
         @return filtered node
      """
      start_node = None
      self.ast = ast
      try:
        start_node = self.filter(self.ast)
        if mutator_opt_arg:
           self.mutatorFunction(start_node, mutator_opt_arg)
        else:
           self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node
 
   def filter_iterator(self, ast):
      """ Calls an iterable filter

      """
      raise NotImplemented

   def apply_all(self, ast, mutator_opt_arg = None):
      """ Apply mutation to all matches 

          @return pointer to last applied mutation
      """
      start_node = None
      self.ast = ast
      try:
         for elem in self.filter_iterator(ast):
            if mutator_opt_arg:
               start_node = self.mutatorFunction(start_node, mutator_opt_arg)
            else:
               start_node = self.mutatorFunction(start_node)
      except NodeNotFound as nf:
         print str(nf)
      return start_node

   def fast_filter(self, ast):
      raise NotImplemented


   def fast_apply_all(self, ast):
      """ Apply mutation to all matches ignoring syntactic order


         @return pointer to last applied mutation
      """
      start_node = None
      self.ast = ast
      for elem in self.fast_filter(ast):
        start_node = self.mutatorFunction(elem)
      return start_node

   def _get_dict_from_clauses(self, clauses, ast, init = None):
      """ Return a dict of clauses from a list of OmpClause objects
         
           Example: [OmpClause('REDUCTION', ...), OmpClause('PRIVATE', ...)]
             will return:  {'REDUCTION' : [....] , 'PRIVATE' : [...]}

         @return dict with clauses
      """
      clause_names = ['SHARED', 'PRIVATE', 'NOWAIT', 'REDUCTION', 'COPY_IN', 'COPY_OUT']
      clause_dict = {}
      if not init:
         clause_dict = self._clauses
      # Note: Each identifiers is a ParamList
      for elem in clauses:

         if not clause_dict.has_key(elem.name):
               clause_dict[elem.name] = []
         
         if elem.name in ['SHARED', 'PRIVATE', 'REDUCTION', 'COPY_IN', 'COPY_OUT']:
            for id in elem.identifiers.params:
               decl = decl_of_id(id, ast)
               if not decl:
                  raise CudaMutatorError(" Declaration of " + id.name + " in " + elem.name + " clause could not be found ")
               # If a declaration with the same name is already stored, pass. Otherwise, append it to the list
               for stored_decl in clause_dict[elem.name]:
                  if decl.name == stored_decl.name:
                     break
               else:
                  clause_dict[elem.name].append(decl)
         elif elem.name == 'NOWAIT':
            clause_dict[elem.name] = True
 
      for name in clause_names:
         if not clause_dict.has_key(name):
            clause_dict[name] = []

      if not init:
         self._clauses = clause_dict
      return  clause_dict

