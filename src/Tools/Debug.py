


from Visitors.dot_visitor import DotWriter
import subprocess
import os
from cStringIO import StringIO

import config


class DotDebugTool(object):
      """ Shows the tree with the highlight nodes selected """
      def __init__(self, highlight = None):
         self.tmpfile = '/tmp/dotfile.dot'
         self.MAX_LINES = 2000
         if type(highlight) != type([]):
            self.highlight = [highlight]
         else:
            self.highlight = highlight

      def apply(self,node):
         if type(node) == type([]):
            for elem in node:
               self.debug_node(elem)
         else:
            self.debug_node(node)

      def debug_node(self, node):
         DotWriter(filename = self.tmpfile, highlight = self.highlight).visit(node)
         size =  len(open(self.tmpfile).readlines())
 #        print str(type(node)) + " --> " + str(size)
         if not (size > self.MAX_LINES):
            p = subprocess.Popen("python " + config.WORKDIR + "/xdot.py " + "/tmp/dotfile.dot", shell=True)
            sts = os.waitpid(p.pid, 0)[1]
         else:
            print "DotDebugTool:::: AST Too big to show"
