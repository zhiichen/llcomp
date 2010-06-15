from pycparser import c_parser, c_ast
from Visitors.generic_visitors import IDFilter, FuncCallFilter, FuncDeclOfNameFilter, OmpForFilter, OmpParallelFilter,  OmpParallelForFilter, FilterError, TypedefFilter, IdentifierTypeFilter
from Tools.tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.search import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Tools.Parse import parse_source
from Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator
from Mutators.AbstractMutator import IgnoreMutationException, AbstractMutator

#from string import Template
#from mako.template import Template
from TemplateEngine.TemplateParser import TemplateParser

import subprocess
from cStringIO import StringIO
from Visitors.clone_visitor import CWriter
import cStringIO


# Copy substructures
import copy

class CudaMutatorError(Exception):
   def __init__(self, description):
      self.description = description

   def __str__(self):
      return "CudaMutatorError :: " + self.description



class AbstractCudaMutator(AbstractMutator):
   def __init__(self, clauses = {}, kernel_name = 'loopKernel', kernel_prefix = ''):
      # BUG: Don't work with optimize
      # self.template_parser = c_parser.CParser(lex_optimize = False, yacc_optimize = False)
      self.kernel_name = kernel_name
      self.kernel_prefix = kernel_prefix
      self._func_def = None
      self._parallel = None
      self._clauses = clauses
      self.device = "cuda"

   # TODO: Clean this function, it has an strange behaviour
   # TODO: Change function name
   def get_names(self, elem, ast):
      """ Return a list of names for a type """
      type = type_of_id(elem, ast)
      while not hasattr(type, 'names') and not hasattr(type, 'name'):
         # TODO: Launch exception if not type attribute
         type = type.type
      if hasattr(type, 'names'):
         return type.names
      else:
         return [type.name]

   def getThreadNum(self, node):
      """ Gets the maximum number of threads needed """
      if node.op == '<' or node.op == '<=':
          return node.right
      else:
          return node.left

   def parse_snippet(self, template_code, subs_dir, name, show = False):
      subtree = None
      if subs_dir:
         template_code = TemplateParser(template_code).render(**subs_dir)
         if show:
            print " Template " + str(template_code)
      try:
         subtree = parse_source(template_code, name)
      except c_parser.ParseError, e:
         print "Parse error:" + str(e)

	 return None
      except IOError:
         print "Pipe Error"
	 return None
      return subtree;

   def _get_dict_from_clauses(self, clauses, ast, init = None):
      """ Return a dict of clauses from a list of OmpClause objects
         
           Example: [OmpClause('REDUCTION', ...), OmpClause('PRIVATE', ...)]
             will return:  {'REDUCTION' : [....] , 'PRIVATE' : [...]}
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
               clause_dict[elem.name].append(decl)
         elif elem.name == 'NOWAIT':
            clause_dict[elem.name] = True
 
      for name in clause_names:
         if not clause_dict.has_key(name):
            clause_dict[name] = []

      if not init:
         self._clauses = clause_dict
      return  clause_dict

   def get_template_array(self, var_list, ast, func = lambda elem : True, name_func = lambda elem : elem.name, type_func = lambda elem : elem.type):
      """ Prepare template array for vars """
      names = []
      def fast_write(elem):
         writeIO = cStringIO.StringIO()
         cw = CWriter(stream = writeIO)
         cw.visit(elem)
         return writeIO.getvalue()

      def decl_write(elem):
         tmp = " ".join(['%s'%stor for stor in elem.storage]) 
         if isinstance(elem.type, c_ast.ArrayDecl):
            tmp += " "
            tmp_node = elem.type
            while not (isinstance(tmp_node, c_ast.TypeDecl) or isinstance(tmp_node, c_ast.PtrDecl)) and tmp_node:
               tmp_node = tmp_node.type

            tmp += fast_write(tmp_node)
 #           tmp += fast_write(elem.type)
 #           self.visit_ArrayDecl(node = node.type, node_name = decl_name, offset = new_offset)
         else:
            tmp += " ".join(['%s'%qual for qual in elem.quals])
            tmp += " "
            tmp += fast_write(elem.type)
         return tmp

      for elem in var_list:
         if func(elem):
            typestr = decl_write(elem)
            # Type string | var name | pointer to type | pointer to var | declaration string
            names.append([typestr, name_func(elem), type_func(elem), elem, fast_write(elem).replace(';','').replace('\n','') ])
      return names
 
   def buildDeclarations(self, numThreads, reduction_node_list, shared_node_list, ast):
      """ Builds the declaration section 
          @param numThreads number of threads
          @return Declarations subtree
      """ 
      # Position in the template for dimA declaration, just in case we change it
      DIMA_POS = 0
      MEMSIZE_POS = 4
      # TODO : Move this array creation to a template filter (something like |type)
      reduction_vars = self.get_template_array(reduction_node_list, ast)
      def check_array(elem):
         return isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem, c_ast.Struct)
      shared_vars = self.get_template_array(shared_node_list, ast, func = check_array) 

      template_code = """
              /* Kernel configuration */
       void kernel_func() {
              int dimA = 1;
              int numThreadsPerBlock = CUDA_NUM_THREADS;
              int numBlocks = dimA / numThreadsPerBlock + (dimA % numThreadsPerBlock?1:0);
              int numElems = numBlocks * numThreadsPerBlock;
              int memSize = numElems * sizeof(double);

              /* Variable declaration */
              % for var in reduction_names:
                  ${var[0]} * reduction_cu_${var[1]};
              % endfor

              % for var in shared_vars:
                  ${var[0]} * ${var[1]}_cu; 
              % endfor
              /* Initialization */
              % for var in reduction_names:
              cudaMalloc((void **) (&reduction_cu_${var[1]}), numElems * sizeof(${var[0]}));
               /* This may be incorrect in case reduction don't start with 0 or 1 */
              cudaMemset(reduction_cu_${var[1]}, (int) ${var[1]}, numElems * sizeof(${var[0]}));
              % endfor

              % for var in shared_vars:
              ${var[1]}_cu = malloc(numElems * sizeof(${var[0]}));
              cudaMalloc((void **) (&${var[1]}_cu), numElems * sizeof(${var[0]}));
              cudaMemcpy(${var[1]}_cu, ${var[1]}, numElems * sizeof(${var[0]}), cudaMemcpyHostToDevice); 
              % endfor
         }

         """
      kernel_init = self.parse_snippet(template_code, {'reduction_names' : reduction_vars, 'shared_vars' : shared_vars}, name = 'Initialization of ' + self.kernel_name).ext[-1].body
#~    from Tools.Debug import DotDebugTool
#~    DotDebugTool().apply(kernel_init)
      kernel_init.decls[DIMA_POS].init = numThreads
      return kernel_init



   def buildRetrieve(self, reduction_vars, modified_shared_vars, ast = None, shared_vars = None):
      memcpy_lines = []
      # TODO: ******************* IMPORTANT
      #           This code does not follow the correct template use
      # If a shared var is modified inside kernel, we retrieve it from device
      for elem in modified_shared_vars:
         # Only malloc / send if it is a complex type
#         if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
#            memcpy_lines += [elem.name]
         if isinstance(elem.type, c_ast.ArrayDecl): 
            memcpy_lines.append([ elem.name, "sizeof(" + " ".join(self.get_names(elem, ast)) +  ") * " +  elem.type.dim.value])
         elif isinstance(elem.type, c_ast.Struct):
            memcpy_lines.append([elem.name,  "sizeof(" + " ".join(self.get_names(elem, ast)) +  ")"])

      # Template source
      template_code = """
      int fake() {
      % for name in reduction_names:
        cudaMemcpy(reduction_loc_${name}, reduction_cu_${name}, memSize, cudaMemcpyDeviceToHost);
      % endfor

      % for elem in shared_names:
        cudaMemcpy(${elem[0]}, ${elem[0]}_cu, ${elem[1]}, cudaMemcpyDeviceToHost);
      % endfor

      checkCUDAError("memcpy");

      % for elem in shared_names:
        cudaFree(${elem[0]}_cu);
      % endfor
      }
      """ 
      return self.parse_snippet(template_code, {'reduction_names' : reduction_vars, 'shared_names' : memcpy_lines}, name = 'Retrieve', show = True).ext[0].body
      
   def buildKernelLaunch(self, reduction_vars, shared_vars, ast):
       # FIXME : reduction_vars is now an array of declarations
       reduction_var_list = ",".join("reduction_cu_" + elem.name for elem in reduction_vars)
       shared_var_list = [];
      # TODO: ******************* IMPORTANT
      #           This code does not follow the correct template use

       for elem in shared_vars:
         # Only malloc / send if it is a complex type
         elem_type = type_of_id(elem, ast)
         ptr =""
         if isinstance(elem_type, c_ast.ArrayDecl) or isinstance(elem_type, c_ast.Struct): 
            shared_var_list += [ptr + str(elem.name) + "_cu"]
         else:
            shared_var_list += [ptr + str(elem.name)]


       kernel_parameters = " "
       if len(reduction_var_list) > 0:
         kernel_parameters += reduction_var_list
       if len(reduction_var_list) > 0 and len(shared_var_list) > 0:
         kernel_parameters += ","
       if len(shared_var_list) > 0:
         kernel_parameters += ",".join(shared_var_list)

       template_code = """
  	#include "llcomp_cuda.h" 

       int fake() {
              dim3 dimGrid (numBlocks);
     	        dim3 dimBlock (numThreadsPerBlock);

          ${kernelName} <<< dimGrid, dimBlock >>> (${kernelParameters});
       }
       """
       # The last element is the object function
       tree = [ elem for elem in self.parse_snippet(template_code, {'kernelParameters' : kernel_parameters,  'kernelName' : self.kernel_name}, name = 'KernelLaunch').ext  if type(elem) == c_ast.FuncDef  ][-1].body
       return tree


   def buildHostReduction(self, reduction_vars, ast):
      if len(reduction_vars) == 0:
         return c_ast.Compound(stmts = [], decls = [])

      reduction_lines = ""
      free_lines = ""
      for elem in reduction_vars:
         free_lines += "cudaFree(reduction_cu_" + (elem.name) + ");\n"
      # TODO: Add shared vars to free

      wait_lines = "cudaThreadSynchronize();\n";
      # TODO : Use common format
      template_code = """
      int fake() {
      #define LLC_REDUCTION_FUNC(dest, fuente) dest = dest + fuente 
   % for var in reduction_vars:
      ${var} = kernelReduction_${type}(reduction_cu_${var}, numElems, ${var});
   % endfor

      /* By default, omp for has a wait at the end */
      ${wait}
      ${free_lines}
      }
      """
      return self.parse_snippet(template_code, {'reduction_vars' : [elem.name for elem in reduction_vars], 'type' : self.get_names(elem, ast)[0], 'free_lines' : free_lines, 'wait' : wait_lines}, name = 'HostReduction').ext[0].body

   def buildKernel(self, shared_list, private_list, reduction_list, loop, ast):
      """ Build CUDA Kernel code """

      reduction_vars = self.get_template_array(reduction_list, ast)
      # Retrieve list of shared vars and build the array to template parsing
      # TODO: Move this to some kind of template function
      def decls_to_param(elem):
         if isinstance(elem.type, c_ast.ArrayDecl):
            return "*" + elem.name + "_cu"
         return elem.name

      shared_vars = self.get_template_array(shared_list, ast, name_func = decls_to_param) 

      # TODO: Clean (Move this to external function)
      decls_dict = {}
      param_var_list = []
      for elem in shared_vars:
         try:
            identifier_type = IdentifierTypeFilter().apply(elem[2])
            if not identifier_type.names[0] in decls_dict:
               typedef_node = TypedefFilter(name = identifier_type.names[0]).apply(ast)
               # TODO: Avoid construction/destruction
               typedefIO = cStringIO.StringIO()
               cw = CWriter(stream = typedefIO)
               cw.visit(typedef_node)
               decls_dict[identifier_type.names[0]] = str(typedefIO.getvalue())
         except NodeNotFound as nnf:
            # It is not a complex type
#            print " Not a userdefined-type " + elem[1]
#            structIO = cStringIO.StringIO()
#            cw = CWriter(stream = structIO)
#            cw.visit(elem[3])
#            param_var_list.append(str(structIO.getvalue()).replace(';','') )
            pass

      typedef_list = [ elem for elem in decls_dict.values() ]

#      print "Typedef :" + str(typedef_list)
#      print "Param_var_list :" + str(param_var_list)
#      print " Reduction_vars : " + str(reduction_vars)
#      print " Shared_vars : " + str(shared_vars)
#
      template_code = """

         #include "llcomp_cuda.h" 
         %for line in typedefs:
            ${line}
         %endfor

/*            # Type string | var name | pointer to type | pointer to var | declaration string */
          __global__ void ${kernelName} (
              ${', '.join( var[0] + "* reduction_cu_" + var[1] for var in reduction_vars)}
              %if len(reduction_vars) > 0 and len(shared_vars) > 0:
                  ,
              %endif 
              ${', '.join( var[0] + var[1] for var in shared_vars)}
         )
          {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          ;
          }
          """
      tree = self.parse_snippet(template_code, {'kernelName' : self.kernel_name, 'reduction_vars' : reduction_vars, 'shared_vars' : shared_vars, 'typedefs' : typedef_list} , name = 'KernelBuild', show = False)

      # OpenMP shared vars are parameters of the kernel function
      if shared_list:
        for elem in shared_list:
            # Replace the name of the declaration in the kernel code. 
            if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
               mut = IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name + '_cu'))
               mut.apply_all(loop.stmt)

      DeclsToParamsMutator().apply(tree.ext[-1].function.decl.type.args)
      # OpenMP Private vars need to be declared inside kernel
      #    - we build a tmp Compound to group all declarations, and insert them once
      tmp = c_ast.Compound(decls= private_list, stmts=[])
      #    - Insert tool removes the parent node of the inserted subtree
      InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')

      # Add the loop statements, (but not the reduction)
      IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx')).apply_all(loop.stmt)
      # Add the loop statements, (but not the reduction)
      IDNameMutator(old = loop.init.lvalue, new = c_ast.ID('idx')).apply_all(loop.cond)

      # Identify function calls inside kernel and replace the definitions to __device__ 
      try:
         for func_call in FuncCallFilter().iterate(loop.stmt):
           print " Writing " + func_call.name.name + " to device "
           try:
              fcm = FuncToDeviceMutator(func_call = func_call).apply(ast)
           except IgnoreMutationException as ime:
              # This function is already implemented on device, so we continue we don't need to convert it
              print "CudaMutator:: Warning :: " + str(ime)
      except NodeNotFound:
         # There are not function calls on the loop.stmt
         pass
      except FilterError as fe:
         raise CudaMutatorError(fe.get_description())
      # Identify function calls inside kernel and replace the definitions to __device__ 
      # TODO: This is incorrect, we should write a subtree instead of a bare string...
      for elem in reduction_list:
         IDNameMutator(old = c_ast.ID(name = elem.name, parent = elem.parent), new = c_ast.ID(name = 'reduction_cu_' + str(elem.name) + '[idx]', parent = elem.parent)).apply_all(loop.stmt)
      # Insert the code inside kernel
      # We need to check if the idx is inside for limits (in case we have more threads than iterations)
      check_boundary_node = c_ast.Compound(decls = None, stmts = [c_ast.If(cond = loop.cond, iftrue = loop.stmt, iffalse = None)], parent = tree.ext[-1].function.body)
      InsertTool(subtree = check_boundary_node, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
      return c_ast.FileAST(ext = [tree.ext[-1]])


class CM_OmpParallelFor(AbstractCudaMutator):
   """ This  mutator locates a omp parallel for reduction, and then
      translate the original source to an equivalent cuda implementation 
   """
   def __init__(self, clauses = {}, kernel_name = 'loopKernel', kernel_prefix = ''):
      """ Constructor """
      self._ompFor = None
      super(CM_OmpParallelFor, self).__init__(clauses, kernel_name, kernel_prefix)


   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      # Build a visitor , matching the OmpFor node of the AST
      f = OmpParallelForFilter()
      node = f.apply(ast)
      self._func_def = f.get_func_def()
      self._parallel = f.get_parallel()
      return node

   def apply_all(self, ast):
      """ Apply mutation to all matches """
      start_node = None
      self.ast = ast
      f = OmpParallelForFilter()
      num = 0;
      try:
         for elem in f.iterate(ast):
            # Save previous state
            old_name = self.kernel_name
            old_clauses = self._clauses
            self.kernel_name = self.kernel_name + str(num)
            # Current scope variables
            self._func_def = f.get_func_def()
            self._parallel = f.get_parallel()
            self._ompFor = f.get_ompFor()
            print str(elem)
            print str(self._ompFor)
            start_node = self.mutatorFunction(ast, self._ompFor)
            # Restore previous state
            self.kernel_name = old_name
            self._clauses = old_clauses
            num += 1;
      except NodeNotFound as nf:
         print str(nf)
      except StopIteration:
         return self._parallel
      return start_node




   def mutatorFunction(self, ast, ompFor_node):
      """ CUDA mutator, writes the for as a kernel
      """
      # Look up a For node which previous brother is the start_node
#      filter = FilterVisitor(match_node_type = c_astFor, prev_brother = ompFor_node)
      # parallelFor = ompFor_node.stmt
      # Parent of the node
      # parent_stmt = self._parallel
      # container_func = self._func_def
      # Maximum number of parallel threads
#      from Tools.Debug import DotDebugTool


      # TODO:  Implement parallel for 
      #             The code after this raise is currently broken, due to changes on template build system
      #              It is an easy fix, but we have no time now.

      maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)

#      DotDebugTool(select_node = maxThreadNumber_node).apply(parallelFor.cond)

      ##################### Statement for cuda
      cuda_stmts = c_ast.Compound(stmts = [], decls = []);

      ##################### Cuda parameters on host

      clause_dict = self._get_dict_from_clauses(self._parallel.clauses + ompFor_node.clauses,  ast)
      shared_params = clause_dict['SHARED']
      private_params = clause_dict['PRIVATE']
      reduction_params = clause_dict['REDUCTION']
      nowait = clause_dict.has_key('NOWAIT')

      ##################### Declarations

      kernel_init_subtree = self.buildDeclarations(numThreads = maxThreadNumber_node, reduction_node_list = reduction_params, shared_node_list = shared_params, ast = ast)
      InsertTool(subtree = kernel_init_subtree, position = "begin").apply(cuda_stmts, 'decls')
           

      ##################### Cuda Kernel 

      # Kernel
      kernel_subtree = self.buildKernel(shared_list = shared_params, 
                        private_list = private_params, 
                        reduction_list = reduction_params,
                        loop = ompFor_node.stmt, ast = ast)

      # Function declaration
      # - Build a node without body
#      tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
#      tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
#      kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
      
#      # Find container function

#      InsertTool(subtree = kernel_decl, position = "begin", node = self._func_def).apply(ast, 'ext')
      # Function definition
      InsertTool(subtree = kernel_subtree, position = "begin" ).apply(ast, 'ext')


      ##################### Loop substitution 
   

      # Kernel Launch
      kernelLaunch_subtree = self.buildKernelLaunch(reduction_vars = reduction_params, shared_vars = shared_params, ast = ompFor_node)
      InsertTool(subtree = kernelLaunch_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Retrieve data
      retrieve_subtree = self.buildRetrieve(reduction_vars = reduction_params, modified_shared_vars = shared_params)
      InsertTool(subtree = retrieve_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Host reduction
      reduction_subtree = self.buildHostReduction(reduction_vars = reduction_params, ast = ast)
      InsertTool(subtree = reduction_subtree, position = "end").apply(cuda_stmts, 'stmts')
      # Replace the entire pragma by a CompoundStatement with all the new statements
      ReplaceTool(new_node = cuda_stmts, old_node = self._parallel.parent).apply(self._parallel.parent.parent, 'stmts')


      ##################### Final tree operations

      # Remove the pragma from the destination code
#      RemoveTool(target_node = self._parallel).apply(container_func, 'stmts')



   def apply(self, ast):
      """ Apply the mutation """
      start_node = None
      try: 
         start_node = self.filter(ast)
         self.mutatorFunction(ast, start_node)
         # Remove pragma from code
      except NodeNotFound as nf:
         print nf
      return ast




