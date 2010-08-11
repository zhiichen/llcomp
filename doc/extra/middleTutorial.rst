|llCoMP| Mutator tutorial
==================================


This tutorial will show step by step how to write a mutator for |llCoMP|. 
This mutator will look for the construct **llc interchange**, and swap
the inmediatly following loops.


Loop Interchange
****************************

In compiler theory, loop interchange is the process of exchanging the order of two iteration variables.

For example, in the code fragment ::

  for(i = 0; i < 10; i++)
    for (j = 0; j < 100; j++)
     a[i][j] = 3.0

loop interchange would result in ::

     for (j = 0; j < 100; j++)
        for(i = 0; i < 10; i++)
            a[i][j] = 3.0

This transformation may lead to performance improvements in some architectures due to improved cache usage, or it
ease other optimizations.

Note that not all iteration variables may be interchanged, due to dependency problems. We will assume that user
have previously resolved dependency problems.

New language construct
****************************

In order to apply this mutator, we need to add a new construct to |llc|. Our intention is that, when 
the user detects a situation where loop interchange may be suitable, she only needs to write this new construct
and |llCoMP| will handle the rest. As an example::

     #pragma llc interchange
     for(i = 0; i < 10; i++)
        for (j = 0; j < 100; j++)
            a[i][j] = 3.0


Note: We are designing a new frontend structure, and probabily the files that need to be edited may change.

More information of this process can be found in the PLY documentation.

In order to add this construct, we need to edit the _c_ast.yaml file, from the pycparser distribution. Navigate to the |llc| section, and
add the llcInterchange construct. This will create a node object, representing the compiler token. Between brackets we can define aditional attributes to the token. *name* is the internal name of the token, used for debugging. *loop* is a link to the first loop of the nested loops that we are going to swap ::

   # llc 
   llcNestedFor : [name, loop*]
   llcInterchange : [name, loop*]
   # Construct
   OmpParallel : [name, clauses**, stmt*]
   OmpFor : [name, clauses**, stmt*]


Next, we need to add the lexical rules for this token, so, open the c_lexer.py file from the pycparser distribution, and navigate to the |llc| section, about line 170. Here you can add token names. In this case, we add the 'INTERCHANGE' token ::

   'LLC', # llc
   # Constructs
   'NESTED_FOR', # nested for
   'INTERCHANGE', # interchange
   ####################
   # OpenMP 3.0 syntax
   'OMP', # openmp
   # Constructs
   'PARALLEL',
   'SECTIONS',


Now, move to the next |llc| section, about line 291. This contain token
definitions, regular expressions that recognized the tokens. This definitions
are defined as python methods. For our token, we define a function called
*t_directive_INTERCHANGE* (directive is the name of the lexical status, and
INTERCHANGE is the name of the token). The second line, beginning with r, is
the regular expression matching the token. Both  *pragma llc interchange* or
*pragma llc swap* recognized as INTERCHANGE. ::

     def t_directive_INTERCHANGE(self, t):
        r'interchange|swap'
        t.type = 'INTERCHANGE'
        return t

Finally, we need to add a parser rule to tell the fronted how this new construct is written. 
open the file c_parser.py from the pycparser distribution, and navigate to the |llc| section.
The grammar rules for |llc| tokens are about line 421. You can add your new rule by writing a 
new method. INTERCHANGE token must precede a for loop, so a iteration-statement
follows the INTERCHANGE token. We save the loop on the loop attribute of the
llcInterchange node.::

    directive_1(self, p):
       """ llc_directive : NESTED_FOR PPHASH PRAGMA OMP workshare_directive"""
       p[0] = c_ast.llcNestedFor(name = 'NESTED FOR', loop = p[5], coord=self._coord(p.lineno(1)))

    def p_llc_directive_2(self, p):
       """ llc_directive : INTERCHANGE iteration_statement"""
       p[0] = c_ast.llcInterchange(name = 'INTERCHANGE', loop = p[2], coord=self._coord(p.lineno(1)))



Now you need to rebuild the compiler tables, so the new grammar rules are build. ::

      # Build the token classes
      $ cd pycparser/
      $ python _ast_gen.py
      # Rebuild tables
      $ python _build_tables.py

To check that you haven't broke anything, move to the |llCoMP| directory and run all tests ::

      $ cd llcomp/
      $ python tests/checkall.py



To check your new construct, create an example file and see if it is parsed properly. The easiest
way is to edit the z_test.py file inside pycparser, adding the new token to the example.




Writing a new mutator
****************************

The easiest way to add a new mutator is to put it on the :mod:`MiddleEnd` of the compiler. 
We will create a new module inside :mod:`MiddleEnd` called Loop, where we will store
different loop optimizations. All modules of |llCoMP| have a similar structure. ::

   Loop/
   ├── __init__.py
   ├── Visitors
   ├── Writers
   ├── tests
   └── Mutators
       ├── __init__.py
       └── LoopInterchange.py
   

The LoopInterchange file will contain all methods and classes required to implement the Loop Interchange Mutator.

First step to write a :term:`Mutator` is to implement a :term:`Filter`. In our case, the :term:`Filter` will look
for the *llcInterchange*  node in the Internal Representation. All filters inherit from the :class:`GenericFilterVisitor`.
The Loop Interchange Filter is easy to implement ::

  class LoopInterchangeFilter(GenericFilterVisitor):
      """ Returns llcInterchange nodes
      """
      def __init__(self):
          def condition(node):
              if type(node) == c_ast.llcInterchange: 
                  return True
              return False
          super(LoopInterchangeFilter, self).__init__(condition_func = condition)
  


The condition of the filter is that the node being check is a llcInterchange node. 
When the filter is running and the condition is True, the current node is returned. 
If our filter does not require an specific order of search, faster search methods can be used.

The :term:`Mutator` for LoopInterchange inherits from :class:`AbstractMutator`. All  mutators share a common structure. The example show a mutator that does nothing but search for the llcInterchange node and return it. ::

 class LoopInterchange(AbstractMutator):
    """ Apply Loop Interchange """ 
    def __init__(self, *args, **kwargs):
        super(LoopInterchange, self).__init__()

    def filter(self, ast):
        """  """
        raise NotImplemented

    def filter_iterator(self, ast):
        """ Fast filter  """
        return NotImplemented

    def fast_filter(self, ast):
        """ Fast filter , looking for binary expressions """
        return LoopInterchangeFilter().dfs_iter(ast)

    def mutatorFunction(self, ast):
        """ Mutator code """
        return ast

 
The fast_filter method calls the Deep First Search iterator of the LoopInterchange filter (note that this have been inherited,
and no additional effort was needed).
For simplicity, we will assume that the construct only appears on a Compound Statement node. 
The ast param is the node returned by the filter, so now it contains a loopInterchange node.
Applying loop interchange to a loop is as easy as implementing this mutator method ::

    # Name things to ease code reading
    interchange_parent = ast.parent.parent  # Note that the parent of any llc node is a pragma node
    interchange_node = ast
    first_loop = ast.loop
    second_loop = ast.loop.stmt
    # 1. Put second loop as the first
    ReplaceTool(new_node = second_loop, old_node = interchange_node.parent).apply(interchange_parent, 'stmts')
    # 2. Preserve parent link
    second_loop.parent = interchange_parent
    # 3. Move the contents of the loop to the first loop
    first_loop.stmt = second_loop.stmt
    first_loop.parent = second_loop
    # Change the new outer loop statements to the new inner loop
    second_loop.stmt = first_loop
    # Return the new outer loop
    return second_loop


In order to ease further compiler phases, all mutators must remove any constructs from the Internal Representation.
It is also a good practice to ensure that the parent node is preserved after applying the mutator. The easiest
way is to call the method link_all_parents , but it is not efficient. We recommend to manually update the parent links.
Future release will make |llCoMP| tools to preserve this links, but currently cannot be guaranteed.

Check the correct implementation of interchange by writing a simple script. Copy the c2c.py example to a new file
called c2interchange.py, and apply the mutator in the Second Layer section. ::

    ###################### Second Layer  : Transformation tools
    # Optimize code
    from MiddleEnd.Loop.Mutators.LoopInterchange import LoopInterchange
    LoopInterchange(start_ast = new_ast).fast_apply_all(new_ast)
    
    

If you run the new script , you see how the loops swaps ::

    $ python bin/c2interchange.py examples/nestedLoop.c output.c







