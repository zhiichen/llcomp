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



class CloneVisitor(OffsetNodeVisitor):
	inside_compound = False

	def __init__(self, filename = None):
		self.filename = filename or sys.stdout
		try:
			self.file = open(self.filename, 'w+') 
		except TypeError:
			self.file = sys.stdout

		self.inside_compound = False

	def __del__(self):
		self.file.close()

	def writeLn(self, offset, string):
		self.write(offset, string)
		self.file.write("\n")

	def write(self, offset, string):
		self.file.write(" " * offset + string)

	def debug(self, node):
		print " >>>>> * <<<< "
		node.show()
		print " >>>> " + str(dir(node)) + "<<< "
		print " >>>>> * <<<< "

	def write_blank(self):
		self.file.write(" ")

	def visit_FileAST(self, node, offset = 0):
		for elem in node.children():
			self.visit(elem, offset)

	def visit_FuncDef(self, node, offset = 0):
		self.visit_Decl(node.decl)
		self.visit_Compound(node.body)

	def visit_ParamList(self, node, offset = 0):
		self.write(offset, "(")
		self.generic_visit(node)
		self.write(offset, ")")
		self.write_blank()

	def visit_Compound(self, node, offset = 0):
		self.writeLn(offset, "\n{\n")
		new_offset = offset + 2
		self.inside_compound = True
		for c in node.children():
				self.visit(c, offset = new_offset)
				self.writeLn(0, ";")
		self.inside_compound = False
		self.writeLn(offset, "}\n")

	def visit_Decl(self, node, offset = 0):
		storage = " ".join(['%s'%stor for stor in node.storage]) 
		self.write(offset, storage);
		self.write_blank()

		# In case it's a function, we need to print the parameter list in a different order
		#	so we send the node.name attribute to the children, where will be printed correctly
		if isinstance(node.type, c_ast.FuncDecl):
			self.visit_FuncDecl(node.type, node.name, offset)
		else:
			if isinstance(node.type, c_ast.ArrayDecl):
				# Array declarations are also different, you need to pass the node name attribute
				decl_name = node.name + " ".join(['%s'%qual for qual in node.quals])
				new_offset = offset
				self.visit_ArrayDecl(node = node.type, node_name = decl_name, offset = new_offset)
			else:
				self.visit_TypeDecl(node.type, offset)
				string = node.name + " ".join(['%s'%qual for qual in node.quals])
				self.write(offset, string)
			if node.init:
				self.write_blank()
				self.write(0, "=")
				self.write_blank()
				self.visit_Constant(node.init)
			if not self.inside_compound:
				self.write(0, ";\n")
				self.write_blank()

	def visit_FuncDecl(self, node, name = "None", offset = 0 ):
		self.visit_TypeDecl(node.type, offset = 0)
		self.write(offset, name)
		if node.args:
			self.visit_ParamList(node.args)
		else:
			self.write(offset, "()")
		self.write_blank()

	def visit_IdentifierType(self, node, offset = 0):
		self.generic_visit(node, offset)
		if len(node.names) >= 1:
			self.write(offset, node.names[0])
		self.write_blank()

	def visit_TypeDecl(self, node, offset = 0):
		self.generic_visit(node)
		self.write_blank()

	def visit_FuncCall(self, node, offset = 0):
		self.visit_ID(node.name)
		self.write(offset, "(")
		self.visit_ExprList(node)
		self.write(offset, ")")
		self.write_blank()

	def visit_ID(self, node, offset = 0):
		self.write(offset, node.name)
		self.write_blank()

	def visit_ExprList(self, node, offset = 0):
		if node.args:
			self.generic_visit(node.args, offset)
		self.write_blank()

	def visit_Constant(self, node, offset = 0):
		if node:
			self.write(offset, node.value)
		# self.write_blank()


	# ******************** Loops ********************
	def visit_For(self, node, offset = 0):
		self.write(offset, "for (")
		self.visit(node.init)
		self.write(0, ";")
		self.write_blank()
		self.visit(node.cond)
		self.write(0, ";")
		self.write_blank()
		self.visit(node.next)
		self.write(0, ")")
		self.visit_Compound(node.stmt, offset)
		self.write_blank()


	# ******************** Expressions ********************
	def visit_Assignment(self, node, offset = 0):
		self.visit(node.lvalue)
		self.write(0, node.op)
		self.write_blank()
		self.visit(node.rvalue, offset = 0)
		# self.write_blank()

	def visit_BinaryOp(self, node, offset = 0):
		self.visit(node.left)
		self.write_blank()
		self.write(0, node.op)
		self.write_blank()
		self.visit(node.right)
		# self.write_blank()

	def visit_UnaryOp(self, node, offset = 0):
		# self.debug(node)
		# node.op is like p-- (always a p), we need to remove it
		self.visit(node.expr)
		self.write(0, node.op[1:])
		self.write_blank()


	# ******************** Array ********************
	def visit_ArrayDecl(self, node, node_name, offset = 0):
		self.visit_TypeDecl(node.type)
		self.write(0, node_name)
		self.write(0, "[")
		if node.dim:
			self.visit_Constant(node.dim)
		self.write(0, "]")
		self.write_blank()

	def visit_ArrayRef(self, node, offset = 0):
		self.visit(node.name)
		self.write(0, "[")
		if node.subscript:
			self.visit(node.subscript, offset)
		self.write(0, "]")
		self.write_blank()


