


from Visitors.dot_visitor import DotWriter
import subprocess
import os
from cStringIO import StringIO




class DotDebugTool(object):
      def __init__(self, select_node = None):
         self.tmpfile = '/tmp/dotfile.dot'
         self.MAX_LINES = 2000
         self.select_node = [select_node]

      def apply(self,node):
         if type(node) == type([]):
            for elem in node:
               self.debug_node(elem)
         else:
            self.debug_node(node)

      def debug_node(self, node):
         DotWriter(filename = self.tmpfile, highlight = self.select_node).visit(node)
         size =  len(open(self.tmpfile).readlines())
         print str(type(node)) + " --> " + str(size)
         if not (size > self.MAX_LINES):
            p = subprocess.Popen("python /home/rreyes/llcomp/src/xdot.py " + "/tmp/dotfile.dot", shell=True)
            sts = os.waitpid(p.pid, 0)[1]
