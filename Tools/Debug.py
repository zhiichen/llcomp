from Backends.DotBackend.Writers import DotWriter
import subprocess
import os
from cStringIO import StringIO

import config


class DotDebugTool(object):
        """Display the tree with xdot, highlighting the given nodes 

            :param highlight: Nodes to highlight, default: None

            .. warning::
                    This tool requires xdot in *config.WORKDIR*, and uses a temporary
                    storage for the dot file
        """
        def __init__(self, highlight = None):
            """Instantiates the debug tool
                 
                :param highlight: Nodes to highlight, default: None
            """
            self.tmpfile = '/tmp/dotfile.dot'
            self.MAX_LINES = 2000
            if type(highlight) != type([]):
                self.highlight = [highlight]
            else:
                self.highlight = highlight

        def apply(self,node):
            """Display the given node (or list of nodes)
                 
                :param node: Node or list of nodes to begin display
            """
            if type(node) == type([]):
                for elem in node:
                    self.debug_node(elem)
            else:
                self.debug_node(node)

        def debug_node(self, node):
            """Display the given node (or list of nodes)
                 
                :param node: Node to display
            """
            DotWriter(filename = self.tmpfile, highlight = self.highlight).visit(node)
            size =  len(open(self.tmpfile).readlines())
            if not (size > self.MAX_LINES):
                p = subprocess.Popen("python " + config.WORKDIR + "/xdot.py " + "/tmp/dotfile.dot", shell=True)
                sts = os.waitpid(p.pid, 0)[1]
            else:
                print "DotDebugTool:::: AST Too big to show"
