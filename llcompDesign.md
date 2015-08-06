# Code Translation #



## Introduction ##

llCoMP is a pattern compiler. It is designed to find a specific specific subtree inside a C AST, and then modify it to generate new code.

## Filter ##


A Filter does a search inside an ast, looking for a subtree that matches the given parameters.

Suppose the following code

```
int main(int argc, char * argv[]) {
  double vnd[10][20];
  int i, j;

  i = 0;
  j = 1;

  vnd[i][j] = 3.0;

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      vnd[i][j] = rand();
    }
  }

}
```

Although matrix are supported by C, and their use is widely extended on several computation devices, they are slower than vectors.
If we want to change matrix to vector, we can look for Matrix declarations and access. In order to do this, we can build a filter like this:

```
class MatrixDeclFilter(GenericFilterVisitor):
   """ Returns the first node with the given attribute
   """
   def __init__(self):
      def condition(node):
          if type(node) == c_ast.ArrayDecl and type(node.type) == c_ast.ArrayDecl:
              if not type(node.type.type) == c_ast.ArrayDecl: # It is not a 3D Matrix
                  return True
      return False
      super(MatrixDeclFilter, self).__init__(condition_func = condition)
```

The filter implements different search algorithms. Currently, we have implemented two search methods: A classic visitor through the AST (which respect the syntax of C), and a fast deeph-first-search algorithm.

If we want to look for all ArrayDecl nodes, we can do:

```
 for node in MatrixDeclFilter().iterate(ast):
   node.show()
```

Which will print all the declaration, in the same order as they were written. However, this method is slow, because it repeats all the search for every declaration on the AST. If we don't need to respect the syntactic order of the code, we can transverse the tree in a faster way:

```
 for node in MatrixDeclFilter().dfs_iter(ast):
     node.show()
```


## Mutator ##

A mutator is a class that can modify an AST. Mutators always have at least a filter and a apply method, and are children of _AbstractMutator_.

The following Mutator changes all Matrix declarations, given by the previously explained filter, to Vector declarations. We have defined two filter methods, a simple filter, which returns only one node and then is destroyed, and a iterator filter, that returns the next node every time.


```
class MatrixDeclToPtr(AbstractMutator):
   """ Convert a Matrix Declaration to a dynamic vector """
   def __init__(self, start_ast):
      self.start_ast = start_ast
      super(MatrixDeclToPtr, self).__init__()

   def filter(self, ast):
      """ Filter definition
         Returns the first node matching with the filter"""
      f = MatrixDeclFilter()
      node = f.apply(ast)
      return node

   def filter_iterator(self, ast):
      """ Iterate through matching nodes """
      return MatrixDeclFilter().iterate(ast) 

   def mutatorFunction(self, ast):
      """ Mutator code """
      array1lvl = ast
      array2lvl = ast.type
      # Change type to vector
      array1lvl.type = array2lvl.type
      # Change size to dim2*dim1
      array1lvl.dim = c_ast.BinaryOp(op = '*', left = array2lvl.dim, right = array1lvl.dim, parent = array1lvl)
    
```

If we have an ast (given by pycparse), we can apply the mutation using the following code:

```
MatrixDeclToPtr(start_ast = new_ast).apply_all(new_ast)
```

In case we don't care about the mutator order, we can invoke the fast version like here:

```
MatrixDeclToPtr(start_ast = new_ast).fast_apply_all(new_ast)
```


### Mutator hierarchy ###

![http://yuml.me/542ade17.png](http://yuml.me/542ade17.png)

## Writer ##

When we have done our translation process, we'll need to read the AST to produce the new code. In order to do this, we have created the Writer classes. Giving an AST called ast, we can write it as C code using this line:

```
from Visitors.clone_visitor import CWriter
CWriter(filename = output_file).visit(new_ast)
```

We can build new writers by extending the classic CWriter. For example, we have implemented a OmpWriter (which prints Omp pragmas) and a CUDAWriter (which support the syntax addons of CUDA). Here you see the CUDAWriter as an example.

```
class CUDAWriter(OmpWriter):
   """ Specific CUDA writer """
   def visit_CUDAKernel(self, node, offset = 0):
      """ __device__ __host__ FuncDef """
      if node.type == 'both':
         self.write(offset, "__device__ __host__")
      else:
         self.write(offset, "__" + str(node.type) + "__")
      self.write_blank();
      self.visit(node.function, offset)

   def visit_CUDAKernelCall(self, node, offset = 0):
      """ name <<<grid, block>>> ( parameters ) """
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

```

