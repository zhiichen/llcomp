from pycparser import c_ast
from Tools.Debug import DotDebugTool

from Tools.Tree import NodeNotFound

from Visitors.generic_visitors import GenericFilterVisitor


class OmpThreadPrivateFilter(GenericFilterVisitor):
   """ Returns the first node with a FuncCall
   """

   def __init__(self, prev_brother = None):
       # The condition __doc__ is used as exception information
       def condition(node):
          """ FuncCall """
          return type(node) == c_ast.OmpThreadPrivate
       super(OmpThreadPrivateFilter, self).__init__(condition_func = condition , prev_brother = prev_brother)


class OmpForFilter(GenericFilterVisitor):
   """ Returns a OmpFor node , the parallel container and the function container
   """
   
   def __init__(self, prev_brother = None):
      self._parallel = None
      self._funcdef = None
      def condition(node):
         """ OmpFor filter """
         return type(node) == c_ast.OmpFor
      super(OmpForFilter, self).__init__(condition_func = condition, prev_brother = prev_brother)

   #############################################
   # This could be a little bit tricky. 
   # By defining specific visitor methods for FuncDef and OmpParallel, we can save the last node visited of this types.
   # Giving the fact that the visit is done in syntax order, the last visited node will be the previous (parent) node of the 
   # wanted node.

   def visit_FuncDef(self, node, prev, offset = 1, ignore = []):
      if not self.match:
         self._funcdef = node
      return self.generic_visit(node, offset, ignore)

   def visit_OmpParallel(self, node, prev, offset = 1, ignore = []):
      if not self.match:
         self._parallel = node
      return self.generic_visit(node, offset, ignore)

   def get_parallel(self):
      return self._parallel

   def get_func_def(self):
      return self._funcdef

   def iterate(self, ast):
       """ Iterate through matching nodes """
       visited_nodes = []
       try:
         while 1:
            visited_nodes.append(self.apply(ast, ignore = visited_nodes))
            yield visited_nodes[-1]
       except NodeNotFound:
#         print "   *** Node not found on iterate, will raise StopIteration *** "
#         raise NodeNotFound("Not")
         raise StopIteration


class OmpParallelFilter(GenericFilterVisitor):
   """ Returns a OmpFor node , the parallel container and the function container
   """
 
   def parallel_condition(self, node):
      """ OmpParallel filter """ 
      # TODO : Move this to a separated filter!!!
      if isinstance(node, c_ast.OmpParallel):
        # If we are looking for a specific device, and the pragma doesn't appear,
        #    this is NOT the correct node
        if device and not self._target_device_node:
          return False
        elif device and self._target_device_node:
           if device == self._target_device_node.device:
              # This is the correct node
              return True
           else:
             return False
        # If we dont need a specific device, this node is valid
        return True
      return False

  
   def __init__(self, condition_func = None, prev_brother = None, device = None):
      def parallel_condition(node):
         """ OmpParallel filter """ 
         # TODO : Move this to a separated filter!!!
         if isinstance(node, c_ast.OmpParallel):
           # If we are looking for a specific device, and the pragma doesn't appear,
           #    this is NOT the correct node
           if device and not self._target_device_node:
             return False
           elif device and self._target_device_node:
              if device == self._target_device_node.device:
                 # This is the correct node
                 return True
              else:
                return False
           # If we dont need a specific device, this node is valid
           return True
         return False

      self._parallel = None
      self._funcdef = None
      self.device = device
      self._target_device_node = None
      super(OmpParallelFilter, self).__init__(condition_func = condition_func or parallel_condition, prev_brother = prev_brother)

   #############################################
   # This could be a little bit tricky. 
   # By defining specific visitor methods for FuncDef and OmpParallel, we can save the last node visited of this types.
   # Giving the fact that the visit is done in syntax order, the last visited node will be the previous (parent) node of the 
   # wanted node.

   def visit_OmpTargetDevice(self, node, prev, offset = 1, ignore = []):
      """ Save target device node """
      if self.device and self.device == node.device:
         self._target_device_node = node
      else:
         self._target_device_node = None
      return self.generic_visit(node, offset, ignore)

   def visit_FuncDef(self, node, prev, offset = 1, ignore = []):
      if not self.match:
         self._funcdef = node
      return self.generic_visit(node, offset, ignore)
   
   def get_func_def(self):
      return self._funcdef


class OmpParallelForFilter(OmpParallelFilter):
   """ Returns a omp parallel for construct """

   def __init__(self, prev_brother = None, device = None):
      self._parallel = None
      self._ompFor = None

      def condition(node):
         """ If node is a parallel, check if has only one stmt and is a for """
         if isinstance(node, c_ast.OmpParallel):
            if isinstance(node.clauses[0], c_ast.OmpFor):
               # print "Match :"  + str(node)
               self._parallel = node
               self._ompFor = node.clauses[0]
               return True
            return False
         return False

      super(OmpParallelForFilter, self).__init__(condition_func = self.parallel_condition and condition, prev_brother = prev_brother, device = device)


   def get_parallel(self):
      return self._parallel

   def get_ompFor(self):
      return self._ompFor



