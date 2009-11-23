from pycparser import c_parser, c_ast


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
	def writeLn(self, offset, string):
		print " " * offset + string
	def write(self, offset, string):
		print " " * offset + string,

	def debug(self, node):
		print " >>>>> * <<<< "
		node.show()
		print " >>>> " + str(dir(node)) + "<<< "
		print " >>>>> * <<<< "


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

	def visit_Compound(self, node, offset = 0):
		self.writeLn(offset, "\n{")
		new_offset = offset + 2
		for c in node.children():
				self.visit(c, offset = new_offset)
				self.writeLn(offset, ";")
		self.writeLn(offset, "\n}")

	def visit_Decl(self, node, offset = 0):
		# In case it's a function, we need to print the parameter list in a different order
		#	so we send the node.name attribute to the children, where will be printed correctly
		if isinstance(node.type, c_ast.FuncDecl):
			self.visit_FuncDecl(offset, node.type, node.name)
		elif isinstance(node.type, c_ast.ArrayDecl):
			# Array declarations are also different, you need to pass the node name attribute
			decl_name = node.name + " ".join(['%s'%qual for qual in node.quals]) + " ".join(['%s'%stor for stor in node.storage])
			new_offset = offset
			self.visit_ArrayDecl(node = node.type, node_name = decl_name, offset = new_offset)
		else:
			self.visit_TypeDecl(node.type, offset)
			string = node.name + " ".join(['%s'%qual for qual in node.quals]) + " ".join(['%s'%stor for stor in node.storage])
			self.write(offset, string)

	def visit_FuncDecl(self, node, name = "None", offset = 0 ):
		self.visit_TypeDecl(node.type, offset = 0)
		self.write(offset, name)
		if node.args:
			self.visit_ParamList(node.args)
		else:
			self.write(offset, "()")

	def visit_IdentifierType(self, node, offset = 0):
		self.generic_visit(node, offset)
		if len(node.names) >= 1:
			self.write(offset, node.names[0])

	def visit_TypeDecl(self, node, offset = 0):
		self.generic_visit(node)

	def visit_FuncCall(self, node, offset = 0):
		self.visit_ID(node.name)
		self.write(offset, "(")
		self.visit_ExprList(node)
		self.write(offset, ")")

	def visit_ID(self, node, offset = 0):
		self.write(offset, node.name)

	def visit_ExprList(self, node, offset = 0):
		if node.args:
			self.generic_visit(node.args, offset)

	def visit_Constant(self, node, offset = 0):
		if node:
			self.write(offset, node.value)


	# ******************** Loops ********************
	def visit_For(self, node, offset = 0):
		self.write(offset, "for (")
		self.visit(node.init)
		self.write(0, ";")
		self.visit(node.cond)
		self.write(0, ";")
		self.visit(node.next)
		self.write(0, ")")
		self.visit_Compound(node.stmt, offset)

	# ******************** Expressions ********************
	def visit_Assignment(self, node, offset = 0):
		self.visit(node.lvalue)
		self.write(offset, node.op)
		self.visit(node.rvalue)

	def visit_BinaryOp(self, node, offset = 0):
		self.visit(node.left)
		self.write(offset, node.op)
		self.visit(node.right)

	def visit_UnaryOp(self, node, offset = 0):
		# self.debug(node)
		# node.op is like p-- (always a p), we need to remove it
		self.visit(node.expr)
		self.write(offset, node.op[1:]);


	# ******************** Array ********************
	def visit_ArrayDecl(self, node, node_name, offset = 0):
		self.visit_TypeDecl(node.type)
		self.write(offset, node_name)
		self.write(offset, "[")
		self.visit_Constant(node.dim)
		self.write(offset, "]")

	def visit_ArrayRef(self, node, offset = 0):
		self.visit(node.name)
		self.write(offset, "[")
		self.visit(node.subscript)
		self.write(offset, "]")


