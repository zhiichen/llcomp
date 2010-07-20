from pycparser import c_parser, c_ast


from Backends.Cuda.Visitors.CM_Visitors import *

from Backends.Common.Visitors.GenericVisitors import *

from Tools.Tree import InsertTool, NodeNotFound, ReplaceTool, RemoveTool
from Tools.Declarations import type_of_id, decl_of_id
from Tools.Dump import Dump
from Tools.Debug import DotDebugTool
from Backends.Common.Mutators.AstSupport import DeclsToParamsMutator, IDNameMutator, FuncToDeviceMutator, PointerMutator


from Backends.Cuda.Mutators.Common import AbstractCudaMutator

from Backends.Cuda.Mutators.CM_OmpParallelFor import CM_OmpParallelFor


from Backends.Common.TemplateEngine.TemplateParser import TemplateParser, get_template_array, get_typedefs_to_template

class CM_llcNestedFor(CM_OmpParallelFor):

    def filter(self, ast):
        """ Filter definition
            Returns the first node matching with the filter"""
        # Build a visitor , matching the OmpFor node of the AST
        f = llcNestedForFilter()
        node = f.apply(ast)
        self._func_def = f.get_func_def()
        self._parallel = f.get_parallel()
        return node

    def apply_all(self, parent_parallel_node, ast):
        """ Apply mutation to all matches """
        start_node = None
        self.ast = ast
        try:
            f = llcNestedForFilter()
            num = 0;
            for elem in f.iterate(ast):
                print " Elem: " + str(elem.loop)
                # Save previous state
                old_name = self.kernel_name
                old_clauses = self._clauses
                self.kernel_name = self.kernel_name + str(num)
                # Current scope variables
                self._func_def = f.get_func_def()
                self._parallel = f.get_parallel()
                # If current node is not child of first parallel node, stop
                if self._parallel != parent_parallel_node:
                    print " Not in current parallel region "
                    from Tools.Debug import DotDebugTool
                    DotDebugTool().apply(self._parallel)
                    raise StopIteration

                print " ** Mutator function ** "
                start_node = self.mutatorFunction(ast, elem.loop)
                print " ** Done ** "
                # Restore previous state
                self.kernel_name = old_name
                self._clauses = old_clauses
                num += 1;
        except NodeNotFound as nf:
            print str(nf)
        except StopIteration:
            return self._parallel
        return start_node


    def buildDeclarations(self, maxLoopA, maxLoopB, reduction_node_list, shared_node_list, ast):
        """ Builds the declaration section 
             @param numThreads number of threads
             @return Declarations subtree
        """ 
        # Position in the template for dimA declaration, just in case we change it
        DIMA_POS = 0
        DIMB_POS = 1
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
                  int dimB = 1;

                  int numThreadsPerBlock = CUDA_NUM_THREADS/2;
                  int numBlocks = dimA / numThreadsPerBlock + (dimA % numThreadsPerBlock?1:0);

                  int numThreadsPerBlockB = CUDA_NUM_THREADS/2;
                  int numBlocksB = dimB / numThreadsPerBlock + (dimB % numThreadsPerBlock?1:0);


                  int numElems = numBlocks * numThreadsPerBlock * numBlocksB * numThreadsPerBlockB;
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
        kernel_init.decls[DIMA_POS].init = maxLoopA
        kernel_init.decls[DIMB_POS].init = maxLoopB
        return kernel_init


    def buildKernel(self, shared_list, private_list, reduction_list, loop, ast):
        """ Build CUDA Kernel code """

        reduction_vars = get_template_array(reduction_list, ast)

        # only for inside for
        loop_list = [loop.init.lvalue.name, loop.stmt.stmts[0].init.lvalue.name]
        private_vars = get_template_array([var for var in private_list if not var.name in loop_list], ast)

        loop_vars = get_template_array([var for var in private_list if var.name in loop_list], ast)

        # Retrieve list of shared vars and build the array to template parsing
        # TODO: Move this to some kind of template function
        def decls_to_param(elem):
            if isinstance(elem.type, c_ast.ArrayDecl):
                return "*" + elem.name + "_cu"
            return elem.name

        shared_vars = get_template_array(shared_list, ast, name_func = decls_to_param) 

        typedef_list = get_typedefs_to_template(shared_vars,ast)

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
             int idx = blockIdx.x * blockDim.x + threadIdx.x;
             int idy = blockIdx.y * blockDim.y + threadIdx.y;

            ${loop_vars[0].declaration} ${loop_vars[0]} = idx;
            ${loop_vars[1].declaration} ${loop_vars[1]} = idy;

             %for var in private_vars:
                ${var.declaration} ${var};
             %endfor
             ;
             }
             """
        tree = self.parse_snippet(template_code, {'kernelName' : self.kernel_name, 'reduction_vars' : reduction_vars, 'shared_vars' : shared_vars, 'typedefs' : typedef_list, 'private_vars' : private_vars, 'loop_vars' : loop_vars} , name = 'KernelBuild', show = True)

        # OpenMP shared vars are parameters of the kernel function
        if shared_list:
          for elem in shared_list:
                # Replace the name of the declaration in the kernel code. 
                if isinstance(elem.type, c_ast.ArrayDecl) or isinstance(elem.type, c_ast.Struct):
                    mut = IDNameMutator(old = c_ast.ID(elem.name), new = c_ast.ID(elem.name + '_cu'))
                    mut.apply_all(loop.stmt)

        DeclsToParamsMutator().apply(tree.ext[-1].function.decl.type.args)
        # OpenMP Private vars need to be declared inside kernel
        #     - we build a tmp Compound to group all declarations, and insert them once
        tmp = c_ast.Compound(decls = [], stmts=[])
        #     - Insert tool removes the parent node of the inserted subtree
        InsertTool(subtree = tmp, position = "end").apply(tree.ext[-1].function.body, 'decls')

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
        # Insert the code inside kernel
        # We need to check if the idx is inside for limits (in case we have more threads than iterations)
        # TODO change this to generic form
        merge_condition_node = c_ast.BinaryOp(op = '&&', left = loop.cond, right = loop.stmt.stmts[0].cond, parent = None)
        # TODO change this to generic form
        check_boundary_node = c_ast.Compound(decls = None, stmts = [c_ast.If(cond = merge_condition_node, iftrue = loop.stmt.stmts[0].stmt, iffalse = None)], parent = tree.ext[-1].function.body)
        # Preserve parent node
        merge_condition_node.parent = check_boundary_node;
        assert check_boundary_node.stmts[0].cond.parent == check_boundary_node

        InsertTool(subtree = check_boundary_node, position = "begin").apply(tree.ext[-1].function.body, 'stmts')
        return c_ast.FileAST(ext = [tree.ext[-1]])



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
                  dim3 dimGrid (numBlocks, numBlocksB);
                    dim3 dimBlock (numThreadsPerBlock, numThreadsPerBlockB);

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


    def buildRetrieve(self, reduction_vars, modified_shared_vars):
        memcpy_lines = ""

        # Template source
        template_code = """
        int fake() {
/*        cudaMemcpy(reduction_loc, reduction_cu, memSize, cudaMemcpyDeviceToHost); */
          ${cudaMemcpyLines}
        checkCUDAError("memcpy");
        }
        """ 

        return self.parse_snippet(template_code, {'cudaMemcpyLines' : memcpy_lines}, name = 'Retrieve').ext[0].body


    def mutatorFunction(self, ast, ompFor_node):
        """ CUDA mutator, writes memory transfer operations for a parallel region
        """

        maxThreadNumber_node = self.getThreadNum(ompFor_node.stmt.cond)
        # TODO: Fix this with general case
        maxLoopB_node = self.getThreadNum(ompFor_node.stmt.stmt.stmts[0].cond)

        ##################### Statement for cuda
        cuda_stmts = c_ast.Compound(stmts = [], decls = []);

        ##################### Cuda parameters on host

        clause_dict = self._get_dict_from_clauses(ompFor_node.clauses,  ast)
  
        reduction_params = clause_dict['REDUCTION']
        nowait = clause_dict.has_key('NOWAIT')
        # Private declarations come from the parent parallel construct
        private_params = clause_dict['PRIVATE']
        shared_params = clause_dict['SHARED']


       
        ##################### Declarations

        kernel_init_subtree = self.buildDeclarations(maxLoopA = maxThreadNumber_node, maxLoopB = maxLoopB_node, reduction_node_list = reduction_params, shared_node_list = [], ast = ast)
        InsertTool(subtree = kernel_init_subtree, position = "begin").apply(cuda_stmts, 'decls')
        

        ##################### Cuda Kernel 

        # Kernel
        kernel_subtree = self.buildKernel(shared_list = shared_params, 
                                private_list = private_params, 
                                reduction_list = reduction_params,
                                loop = ompFor_node.stmt, ast = ast)

        # Function declaration
        # - Build a node without body
#        tmp = c_ast.CUDAKernel(function = copy.deepcopy(kernel_subtree.ext[0].function), type = 'global', name = kernel_subtree.ext[0].name)
#        tmp.function.body = c_ast.Compound(stmts = None, decls = None); # If both of stmts and decls are none, it won't be printed
#        kernel_decl = c_ast.Compound(stmts = [tmp], decls = None)
        
#        # Find container function
#        InsertTool(subtree = kernel_decl, position = "begin", node = self._func_def).apply(ast, 'ext')
        # Function definition
        InsertTool(subtree = kernel_subtree, position = "begin", node = self._func_def ).apply(ast, 'ext')

        # Support subtree
        # support_subtree = self.buildSupport()
        # InsertTool(subtree = c_ast.Compound(stmts = support_subtree.stmts, decls = None), position = "end").apply(ast, 'ext')
        # InsertTool(subtree = c_ast.Compound(decls = support_subtree.decls, stmts = None), position = "begin").apply(ast, 'ext')


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
#        DotDebugTool(highlight = [ompFor_node]).apply(self._parallel)
        if isinstance(ompFor_node.parent.parent, c_ast.Compound) :
            ReplaceTool(new_node = cuda_stmts, old_node = ompFor_node.parent).apply(ompFor_node.parent.parent, 'stmts')
        else:
            # Maybe we have an error here
            ompFor_node.parent.parent.stmt = cuda_stmts




