from pycparser import c_parser, c_ast
import sys

tab_size = 2;

class OffsetNodeVisitor(object):
    def visit(self, node, offset = 0):
        """ Visit a node. 
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, offset)

    def generic_visit(self, node, offset = 0):
        """ Called if no explicit visitor function exists for a 
            node. Implements preorder visiting of the node.
         """
        for c in node.children():
            self.visit(c)



#class CWriter(OffsetNodeVisitor):
#    inside = False
#
#    def __init__(self, filename = None, stream = None):
#        self.filename = filename or sys.stdout
#        if not stream:
#            try:
#                self.file = open(self.filename, 'w+') 
#            except TypeError:
#                self.file = sys.stdout
#        else:
#            try:
#                self.file = stream
#            except TypeError:
#                print "Error: Unknown stream error"
#
#        self.inside = False
#
#    def __del__(self):
#        """ Ensure closing the file when object dissapears """
#        self.file.close()
#
#
#    # ********** Writing support **********
#
#    def writeLn(self, offset, string):
#        self.write(offset, string)
#        self.file.write("\n")
#
#    def write(self, offset, string):
#        self.file.write(" " * offset + string)
#
#    def debug(self, node):
#        print " >>>>> * <<<< "
#        node.show()
#        print " >>>> " + str(dir(node)) + "<<< "
#        print " >>>>> * <<<< "
#
#    def write_blank(self):
#        self.file.write(" ")
#
#    # ********** Support functions **********
#
#    def generic_visit_nodeList(self, nodeList, separator, offset):
#        """ Visit a list of nodes , writing values within a separator """
#        i = 0
#        for i in range(0, len(nodeList) - 1):
#            self.visit(nodeList[i])
#            self.write(offset, separator)
#        self.visit(nodeList[len(nodeList) - 1])
#
#    def visit_FileAST(self, node, offset = 0):
#        for elem in node.children():
#            self.visit(elem, offset)
#            if type(elem) != c_ast.Pragma:
#                self.writeLn(0, ";")
#
#
#    # ********** Grammar **********
#
#    def visit_FuncDef(self, node, offset = 0):
#        self.visit_Decl(node.decl)
#        self.visit_Compound(node.body)
#
#    def visit_ParamList(self, node, offset = 0):
#        self.write(offset, "(")
#        self.generic_visit_nodeList(node.children(), ',', offset)
#        self.write(offset, ")")
#        self.write_blank()
#
#    def visit_Compound(self, node, offset = 0):
#        if (hasattr(node, "stmts") and not node.stmts) and (hasattr(node, "decls") and not node.decls):
#            return
#        self.writeLn(offset, "\n{\n")
#        new_offset = offset + 2
#        self.inside = True
#        for c in node.children():
#            self.visit(c, offset = new_offset)
#            self.writeLn(0, ";")
#        self.inside = False
#        self.writeLn(offset, "}\n")
#
#    def visit_Decl(self, node, offset = 0):
#
#        storage = " ".join(['%s'%stor for stor in node.storage]) 
#        self.write(offset, storage);
#        self.write_blank()
#
#        # In case it's a function, we need to print the parameter list in a different order
#        #    so we send the node.name attribute to the children, where will be printed correctly
#        if isinstance(node.type, c_ast.FuncDecl):
#            self.visit_FuncDecl(node.type, node.name, offset)
#        else:
#            if isinstance(node.type, c_ast.ArrayDecl):
#                # Array declarations are also different, you need to pass the node name attribute
#                decl_name = node.name + " ".join(['%s'%qual for qual in node.quals])
#                new_offset = offset
#                self.write_blank()
#                # ** TODO: Do not perform this search, change recursion order... ** 
#                tmp_node = node.type
#                while not (isinstance(tmp_node, c_ast.TypeDecl) or isinstance(tmp_node, c_ast.PtrDecl)) and tmp_node:
#                    tmp_node = tmp_node.type
#
#                self.visit(tmp_node, offset)
#                self.write(0, decl_name)
#                self.visit_ArrayDecl(node = node.type, node_name = decl_name, offset = new_offset)
#            else:
#                self.write(offset, " ".join(['%s'%qual for qual in node.quals]))
#                self.write_blank()
#                self.visit(node.type, offset) 
#                # Check if node.name is not NoneType
#                if type(node.name) != None.__class__:
#                    self.write(offset, node.name)
#            if hasattr(node, 'init') and node.init:
#                self.write_blank()
#                self.write(0, "=")
#                self.write_blank()
#                if isinstance(node.init, c_ast.ExprList):
#                    self.write(0, '{')
#                    self.visit(node.init)
#                    self.write(0, '}')
#                else:
#                    self.visit(node.init)
#            if not self.inside:
#                self.write(0, ";\n")
#                self.write_blank()
#
#    def visit_FuncDecl(self, node, name = "None", offset = 0 ):
#        self.visit(node.type, offset = 0)
#        self.write(offset, str(name))
#        if node.args:
#            self.inside = 1
#            self.visit_ParamList(node.args)
#            self.inside = 0
#        else:
#            self.write(offset, "()")
#        self.write_blank()
#    
#    def visit_TypeDecl(self, node, offset = 0):
#        self.generic_visit(node)
#        self.write_blank()
#
#    def visit_Typedef(self, node, offset = 0):
#        # A Typedef is almost like any other Decl
#        self.visit_Decl(node, offset)
#
#
#    def visit_PtrDecl(self, node, offset = 0):
#        self.visit(node.type)
#        # self.write(offset, node.quals)
#        self.write(offset, "*")
#
#
#    # ******************** Types ********************
#    def visit_IdentifierType(self, node, offset = 0):
#        self.generic_visit(node, offset)
#        if len(node.names) >= 1:
#            self.write(offset, node.names[0])
#        # self.write_blank()
#
#
#    def visit_FuncCall(self, node, offset = 0):
#        self.visit_ID(node.name)
#        self.write(offset, "(")
#        if node.args:
#            self.visit_ExprList(node.args)
#        else:
#            self.write_blank()
#        self.write(offset, ")")
#        self.write_blank()
#
#    def visit_ID(self, node, offset = 0):
#        self.write(offset, node.name)
#        # self.write_blank()
#
#    def visit_ExprList(self, node, offset = 0):
#        if node.exprs:
#            self.generic_visit_nodeList(node.exprs, ",", offset)
#        self.write_blank()
#
#    def visit_Constant(self, node, offset = 0):
#        if node:
#            self.write(offset, node.value)
#
#    def visit_Break(self, node, offset = 0):
#        self.write(offset, 'break')
#        self.write_blank()
#
#    def visit_Continue(self, node, offset = 0):
#        self.write(offset, 'continue')
#        self.write_blank()
#
#
#    def visit_Return(self, node, offset = 0):
#        self.write(offset, "return")
#        self.write_blank()
#        if node.expr:
#            self.visit(node.expr)
#
#    # ******************** Loops ********************
#    def visit_For(self, node, offset = 0):
#        self.write(offset, "for (")
#        self.visit(node.init)
#        self.write(0, ";")
#        self.write_blank()
#        self.visit(node.cond)
#        self.write(0, ";")
#        self.write_blank()
#        self.visit(node.next)
#        self.write(0, ")")
#        if node.stmt:
#            if type(node.stmt) == c_ast.Compound:
#                self.visit_Compound(node.stmt, offset)
#            else:
#                self.write_blank()
#                self.visit(node.stmt, offset)
#        self.write_blank()
#
#    def visit_While(self, node, offset = 0):
#        self.write(offset, "while (")
#        self.visit(node.cond)
#        self.write(0, ")")
#        self.visit(node.stmt)
#
#    def visit_DoWhile(self, node, offset = 0):
#        self.write(offset, "do")
#        self.visit(node.stmt)
#        self.write(offset, "while (")
#        self.visit(node.cond)
#        self.write(0, ")")
#
#
#    # ******************** Expressions ********************
#    def visit_Assignment(self, node, offset = 0):
#        self.visit(node.lvalue)
#        self.write(0, node.op)
#        self.write_blank()
#        self.visit(node.rvalue, offset = 0)
#        # self.write_blank()
#
#    def visit_TernaryOp(self, node, offset = 0):
#        self.write(offset, '(')
#        self.write(offset, '(')
#        self.visit(node.cond)
#        self.write(0, ')')
#        self.write(0, '?')
#        self.visit(node.iftrue)
#        self.write(0, ':')
#        self.visit(node.iffalse)
#        self.write(0, ')')
#
#    def visit_BinaryOp(self, node, offset = 0):
#        if (isinstance(node.left, c_ast.BinaryOp)):
#            self.write(offset, "(")
#            self.visit(node.left)
#            self.write(offset, ")")
#        else:
#            self.visit(node.left)
#
#        self.write_blank()
#        self.write(0, node.op)
#        self.write_blank()
#        if (isinstance(node.right, c_ast.BinaryOp)):
#            self.write(offset, "(")
#            self.visit(node.right)
#            self.write(offset, ")")
#        else:
#            self.visit(node.right)
#        # self.write_blank()
#
#    def visit_UnaryOp(self, node, offset = 0):
#        if node.op == 'sizeof':
#            # sizeof is considered as an UnaryOp
#            self.write(0, 'sizeof( ')
#            self.visit(node.expr)
#            self.write(0, ')')
#            self.write_blank()
#        elif len(node.op) == 1:
#            self.write(0, node.op)
#            self.visit(node.expr)
#            self.write_blank()
#        else:
#            # node.op is like p-- (always a p), we need to remove it
#            self.visit(node.expr)
#            self.write(0, node.op[1:])
#            self.write_blank()
#
#
#    # ******************** Array ********************
#    def visit_ArrayDecl(self, node, node_name, offset = 0):
#        self.write(0, "[")
#        if node.dim:
#            self.visit(node.dim)
#        self.write(0, "]")
#        self.write_blank()
#        if isinstance(node.type, c_ast.ArrayDecl):
#            self.visit_ArrayDecl(node.type, node_name, offset)
#
#
#    def visit_ArrayRef(self, node, offset = 0):
#        self.visit(node.name)
#        self.write(0, "[")
#        if node.subscript:
#            self.visit(node.subscript, offset)
#        self.write(0, "]")
#        self.write_blank()
#
#
#    # ******************** Struct ********************
#
#    def visit_Struct(self, node, offset = 0):
#        self.write(offset, "struct")
#        self.write_blank()
#        if node.name:
#            self.write(offset, node.name)
#        if type(node.decls) != None.__class__:
#            self.write(0, "{")
#            for elem in node.decls:
#                self.visit(elem, offset)
#            self.write(0, "}")
#        self.write_blank()
#
#
#    def visit_StructRef(self, node, offset = 0):
#        self.visit(node.name)
#        self.write(0, node.type)
#        self.visit(node.field)
#        self.write_blank()
#
#    # ******************** Conditionals ********************
#    def visit_If(self, node, offset = 0):
#        self.write(offset, "if");
#        self.write_blank();
#        self.write(offset, "(");
#        self.visit(node.cond, offset)
#        self.write(offset, ")");
#        self.write_blank();
#        self.visit(node.iftrue, offset);
#        if not isinstance(node.iftrue, c_ast.Compound):
#            self.write(offset, ";")
#        if node.iffalse:
#            self.write(offset, "else");
#            self.write_blank()
#            self.visit(node.iffalse)
#
#    def visit_Switch(self, node, offset = 0):
#        self.write(offset, "switch (")
#        self.visit(node.cond)
#        self.write(0, ")")
#        self.visit(node.stmt)
#
#    def visit_Case(self, node, offset = 0):
#        self.write(offset, "case ")
#        self.visit(node.expr)
#        self.write(0, " : ")
#        self.visit(node.stmt)
#
#    def visit_Default(self, node, offset = 0):
#        self.write(offset, "default ")
#        self.write(0, " : ")
#        self.visit(node.stmt)
#
#
#    # ******************** Typecast ********************
#    def visit_Cast(self, node, offset = 0):
#        self.write(offset, '(')
#        self.visit(node.to_type)
##        self.generic_visit(node.childrens)
#        self.write(offset, ')')
#        self.write(offset, '(')
#        self.visit(node.expr)
#        self.write(offset, ')')
#
#    # ******************** Language Extensions ********************
#    def visit_Pragma(self, node, offset = 0):
#        self.write(offset, "#pragma")
#        self.write_blank();
#        self.write(0, node.name)
#        self.visit(node.stmt)
#    #    self.writeLn(offset, node.name)
#
#class OmpWriter(CWriter):
#    """ OpenMP code writer """
#    def visit_OmpParallel(self, node, offset):
#        self.write_blank();
#        self.write(offset, node.name)
#        self.write_blank();
#        # self.write(offset, 'parallel')
#        # self.write_blank();
#        if node.clauses:
#            for elem in node.clauses:
#                self.visit(elem)
#                self.write_blank();
#        if node.stmt:
#            self.writeLn(offset, "")
#            self.visit(node.stmt)
#        self.write_blank();
#
#    def visit_OmpThreadPrivate(self, node, offset):
#        self.write_blank();
#        self.write(offset, node.name)
#        self.write_blank()
#        if node.identifiers:
#            for elem in node.identifiers.params:
#                self.write(offset, '(')
#                self.visit(elem)
#                self.write(offset, ')')
#                self.write_blank()
#
#    def visit_OmpFor(self, node, offset):
#        self.write_blank();
#        self.write(offset, node.name)
#        self.write_blank();
#        if node.clauses:
#            for elem in node.clauses:
#                self.visit(elem)
#                self.write_blank();
#
#        self.write_blank();
#        # OmpFor always has an stmt 
#        if node.stmt:
#            self.writeLn(offset, " ")
#            self.visit(node.stmt)
#        self.write_blank();  
#
#    def visit_OmpClause(self, node, offset):
#        # Handwrite the case of device clause
#        if node.name == 'cuda':
#            self.write(offset, 'device(' + node.name.lower() + ")")
#        else:
#            self.write(offset, node.name.lower())
#        # TODO: Fix reduction parenthesis error
#        if node.name == 'REDUCTION':
#            self.write(0, '(');
#            self.write(0, node.type)
#            self.write(0, ':')
#
#        if node.identifiers:
#	        self.visit(node.identifiers)
#
#        if node.name == 'REDUCTION':
#            self.write(0, ')');
#
#
#class CUDAWriter(OmpWriter):
#    """ Specific CUDA writer """
#    def visit_CUDAKernel(self, node, offset = 0):
#        if node.type == 'both':
#            self.write(offset, "__device__ __host__")
#        else:
#            self.write(offset, "__" + str(node.type) + "__")
#        self.write_blank();
#        self.visit(node.function, offset)
#
#    def visit_CUDAKernelCall(self, node, offset = 0):
#        self.visit_ID(node.name)
#        self.write(offset, "<<<")
#        self.visit(node.grid)
#        self.write(offset, ",")
#        self.visit(node.block)
#        self.write(offset, ">>>")
#        self.write(offset, "(")
#        if node.args:
#            self.visit_ExprList(node.args)
#        else:
#            self.write_blank()
#        self.write(offset, ")")
#        self.write_blank()
#
