from pycparser import c_ast, c_parser


from Backends.Common.Visitors.GenericVisitors import *


from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.Declarations import type_of_id

from Tools.Debug import DotDebugTool
from Frontend.Parse import parse_source
from Backends.Common.Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator
from Backends.Common.Mutators.AbstractMutator import IgnoreMutationException, AbstractMutator, AbortMutationException

from Backends.Common.TemplateEngine.TemplateParser import TemplateParser, get_template_array, get_typedefs_to_template


import cStringIO

from Backends.C.Writers.CWriter import CWriter

class CudaMutatorError(Exception):
    def __init__(self, description):
        self.description = description

    def __str__(self):
        return "CudaMutatorError :: " + self.description



class AbstractCudaMutator(AbstractMutator):
    """ Common methods to work with CUDA

    """
    def __init__(self, clauses = None, kernel_name = 'loopKernel', kernel_prefix = ''):
        if clauses == None:
            clauses = {}
        self.kernel_name = kernel_name
        self.kernel_prefix = "CM_" + kernel_prefix
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


	 # TODO: Change method name to getLoopSize
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


 
    def buildDeclarations(self, numThreads, reduction_node_list, shared_node_list, ast):
        """ Builds the declaration section 
             @param numThreads number of threads
             @return Declarations subtree
        """ 
        # Position in the template for dimA declaration, just in case we change it
        DIMA_POS = 0
        MEMSIZE_POS = 4
        # TODO : Move this array creation to a template filter (something like |type)
        reduction_vars = get_template_array(reduction_node_list, ast)
        def check_array(elem):
            return isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem, c_ast.Struct)
        shared_vars = get_template_array(shared_node_list, ast, func = check_array) 

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
                        ${var.type} * reduction_cu_${var.name};
                  % endfor

                  % for var in shared_vars:
                        ${var.type} * ${var}_cu; 
                  % endfor
                  /* Initialization */
                  % for var in reduction_names:
                    cudaMalloc((void **) (&reduction_cu_${var.name}), numElems * sizeof(${var.type}));
                    /* This may be incorrect in case reduction don't start with 0 or 1 */
                    cudaMemset(reduction_cu_${var.name}, (int) ${var.name}, numElems * sizeof(${var.type}));
                  % endfor

                  % for var in shared_vars:
                  ${var.name}_cu = malloc(numElems * sizeof(${var.type}));
                  cudaMalloc((void **) (&${var.name}_cu), numElems * sizeof(${var.type}));
                  cudaMemcpy(${var.name}_cu, ${var.name}, numElems * sizeof(${var.type}), cudaMemcpyHostToDevice); 
                  % endfor
            }

            """
        kernel_init = self.parse_snippet(template_code, {'reduction_names' : reduction_vars, 'shared_vars' : shared_vars}, name = 'Initialization of ' + self.kernel_name, show = False).ext[-1].body
        kernel_init.decls[DIMA_POS].init = numThreads
        return kernel_init



    def buildRetrieve(self, reduction_vars, modified_shared_vars, ast = None, shared_vars = None):
        memcpy_lines = []
        reduction_vars = get_template_array(reduction_vars, ast)
        shared_vars     = get_template_array(modified_shared_vars, ast)

        # Template source
        template_code = """
        int fake() {
        % for var in reduction_vars:
          cudaMemcpy(reduction_loc_${var.name}, reduction_cu_${var.name}, memSize, cudaMemcpyDeviceToHost);
        % endfor

        % for var in shared_vars:
          cudaMemcpy(${var.name}, ${var.name}_cu, sizeof(${var.type}) * ${var.numelems}, cudaMemcpyDeviceToHost);
        % endfor

        checkCUDAError("memcpy");

        % for var in shared_vars:
        /*  cudaFree(${var.name}_cu);*/
        % endfor
        }
        """ 
        return self.parse_snippet(template_code, {'reduction_vars' : reduction_vars, 'shared_vars' : shared_vars}, name = 'Retrieve', show = False).ext[0].body
        
    def buildKernelLaunch(self, reduction_vars, shared_vars, ast):
        # TODO: Move this to some kind of template function
        def decls_to_param(elem):
            if isinstance(elem.type, c_ast.ArrayDecl):
                return elem.name + "_cu"
            return elem.name

        shared_vars = get_template_array(shared_vars, ast, name_func = decls_to_param) 

        template_code = """
        #include "llcomp_cuda.h" 
        
         int fake() {
                  dim3 dimGrid (numBlocks);
                    dim3 dimBlock (numThreadsPerBlock);

             ${kernelName} <<< dimGrid, dimBlock >>> (${', '.join("reduction_cu_" + var.name for var in reduction_vars)}
                  %if len(reduction_vars) > 0 and len(shared_vars) > 0:
                        ,
                  %endif 
                  ${', '.join( var.name for var in shared_vars)});
         }
        """
        # The last element is the object function
        tree = [ elem for elem in self.parse_snippet(template_code, {'reduction_vars' : reduction_vars, 'shared_vars' : shared_vars,  'kernelName' : self.kernel_name}, name = 'KernelLaunch', show = False).ext  if type(elem) == c_ast.FuncDef  ][-1].body
        return tree


    def buildHostReduction(self, reduction_vars, ast):
        """ Instanciate the reduction pattern 
    
            @return Compound with the reduction code
        """
        if len(reduction_vars) == 0:
            return c_ast.Compound(stmts = [], decls = [])

        template_code = """
        int fake() {
        #define LLC_REDUCTION_FUNC(dest, fuente) dest = dest + fuente 
        % for var in reduction_vars:
            ${var} = kernelReduction_${var.type}(reduction_cu_${var.name}, numElems, ${var.name});
        % endfor

        /* By default, omp for has a wait at the end */
        % if not nowait:
            cudaThreadSynchronize();
        % endif

        % for var in reduction_vars:
        cudaFree(reduction_cu_${var.name});
        % endfor
        }
        """
        return self.parse_snippet(template_code, {'reduction_vars' : get_template_array(reduction_vars, ast), 'nowait' : False}, name = 'HostReduction').ext[0].body

    def buildKernel(self, shared_list, private_list, reduction_list, loop, ast):
        """ Build CUDA Kernel code """

        reduction_vars = get_template_array(reduction_list, ast)

        # Retrieve list of shared vars and build the array to template parsing
        loop_list = [loop.init.lvalue.name]

        private_vars = get_template_array([var for var in private_list if not var.name in loop_list], ast)
        loop_vars = get_template_array([var for var in private_list if var.name in loop_list], ast)

#~        print "Loop : " + str(loop.init.lvalue.name)
#~        print "Loop_list : " + str(loop_list)
#~        print "*** Private list : " + str(private_list)
#~        print "*** Private vars : " + str(private_vars)
#~        print "*** Loop vars : " + str(loop_vars)

        if len(loop_vars) == 0:
            raise AbortMutationException(" No loop vars found: check your private clause\n")


        # TODO: Move this to some kind of template function
        def decls_to_param(elem):
            if isinstance(elem.type, c_ast.ArrayDecl):
                return "*" + elem.name + "_cu"
            return elem.name

        shared_vars = get_template_array(shared_list, ast, name_func = decls_to_param) 

        clause_vars = []
        clause_vars.extend(shared_vars)
        clause_vars.extend(private_vars)
        typedef_list = get_typedefs_to_template(clause_vars,ast)


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
            print " Filter error "
            raise CudaMutatorError(fe.get_description())
        # Identify function calls inside kernel and replace the definitions to __device__ 
        # TODO: This is incorrect, we should write a subtree instead of a bare string...
        for elem in reduction_list:
            IDNameMutator(old = c_ast.ID(name = elem.name, parent = elem.parent), new = c_ast.ID(name = 'reduction_cu_' + str(elem.name) + '[idx]', parent = elem.parent)).apply_all(loop.stmt)

        # OpenMP shared vars are parameters of the kernel function
        if shared_list:
          for elem in shared_list:
                # Replace the name of the declaration in the kernel code. 
                if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
                    mut = IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name + '_cu'))
                    mut.apply_all(loop.stmt)



        template_code = """

            #include "llcomp_cuda.h" 
            %for line in typedefs:
                ${line}
            %endfor

             __global__ void ${kernelName} ( 
                  ${', '.join(str(var.type) + " * reduction_cu_" + str(var.name) for var in reduction_vars)}
                  %if len(reduction_vars) > 0 and len(shared_vars) > 0:
                        ,
                  %endif 
                  ${', '.join(str(var.type) + " " + str(var.name) for var in shared_vars)}
            )
             {

              ${loop_vars[0].declaration} ${loop_vars[0]}  = blockIdx.x * blockDim.x + threadIdx.x;
 			 %for var in private_vars:
              ${var.declaration} ${var};
          %endfor
                if ((${loop.init.rvalue} < i) && (${loop.cond})) 
                    ${loop.stmt};
                
             }
             """
        tree = self.parse_snippet(template_code, {'kernelName' : self.kernel_name, 'reduction_vars' : reduction_vars, 'shared_vars' : shared_vars, 'typedefs' : typedef_list, 'private_vars' : private_vars, 'loop_vars' : loop_vars, 'loop' : loop} , name = 'KernelBuild', show = True)

        
        # DeclsToParamsMutator().apply(tree.ext[-1].function.decl.type.args)
        # OpenMP Private vars need to be declared inside kernel
        #     - we build a tmp Compound to group all declarations, and insert them once
        tmp = c_ast.Compound(decls= private_list, stmts=[])
        #     - Insert tool removes the parent node of the inserted subtree
        InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')


        # Insert the code inside kernel
        # We need to check if the idx is inside for limits (in case we have more threads than iterations)
#        check_boundary_node = c_ast.Compound(decls = None, stmts = [c_ast.If(cond = loop.cond, iftrue = loop.stmt, iffalse = None)], parent = tree.ext[-1].function.body)
#        InsertTool(subtree = check_boundary_node, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
        return c_ast.FileAST(ext = [tree.ext[-1]])



class CudaTransformer:
    @staticmethod
    def apply(ast):
 #       from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor
        from Backends.Cuda.Mutators.CM_OmpParallel import CM_OmpParallel

        # cuda_ast = CM_OmpParallelFor().apply_all(ast)
        # TODO Need to link parents after this?
        cuda_ast = CM_OmpParallel().apply_all(ast)
        return cuda_ast


