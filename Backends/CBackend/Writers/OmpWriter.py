
from pycparser import c_parser, c_ast

from Backends.CBackend.Writers.CWriter import CWriter

class OmpWriter(CWriter):
    """ Visitor which translates the IR to C/OpenMP.

    """
    def visit_OmpParallel(self, node, offset):
        self.write_blank();
        self.write(offset, node.name)
        self.write_blank();
        # self.write(offset, 'parallel')
        # self.write_blank();
        if node.clauses:
            for elem in node.clauses:
                self.visit(elem)
                self.write_blank();
        if node.stmt:
            self.writeLn(offset, "")
            self.visit(node.stmt)
        self.write_blank();

    def visit_OmpThreadPrivate(self, node, offset):
        self.write_blank();
        self.write(offset, node.name)
        self.write_blank()
        if node.identifiers:
            for elem in node.identifiers.params:
                self.write(offset, '(')
                self.visit(elem)
                self.write(offset, ')')
                self.write_blank()

    def visit_OmpFor(self, node, offset):
        self.write_blank();
        self.write(offset, node.name)
        self.write_blank();
        if node.clauses:
            for elem in node.clauses:
                self.visit(elem)
                self.write_blank();

        self.write_blank();
        # OmpFor always has an stmt 
        if node.stmt:
            self.writeLn(offset, " ")
            self.visit(node.stmt)
        self.write_blank();  

    def visit_OmpClause(self, node, offset):
        # Handwrite the case of device clause
        if node.name == 'cuda':
            self.write(offset, 'device(' + node.name.lower() + ")")
        else:
            self.write(offset, node.name.lower())
        # TODO: Fix reduction parenthesis error
        if node.name == 'REDUCTION':
            self.write(0, '(');
            self.write(0, node.type)
            self.write(0, ':')

        if node.identifiers:
	        self.visit(node.identifiers)

        if node.name == 'REDUCTION':
            self.write(0, ')');


