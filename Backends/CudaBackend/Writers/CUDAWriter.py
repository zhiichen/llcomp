from pycparser import c_ast

from Backends.CBackend.Writers.OmpWriter import OmpWriter



class CUDAWriter(OmpWriter):
    """ CUDA writer """
    def visit_CUDAKernel(self, node, offset = 0):
        if node.type == 'both':
            self.write(offset, "__device__ __host__")
        else:
            self.write(offset, "__" + str(node.type) + "__")
        self.write_blank();
        self.visit(node.function, offset)

    def visit_CUDAKernelCall(self, node, offset = 0):
        self.visit_ID(node.name)
        self.write(offset, "<<<")
        self.visit(node.grid)
        self.write(offset, ",")
        self.visit(node.block)
        self.write(offset, ">>>")
        self.write(offset, "(")
        if node.args:
            self.visit_ExprList(node.args)
        else:
            self.write_blank()
        self.write(offset, ")")
        self.write_blank()

