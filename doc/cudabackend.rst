CUDA Backend
*****************
.. module:: Cuda 
   :synopsis: Converts IR to CUDA


The :mod:`Cuda` module contains a set of Mutators, Filters and Templates
which creates CUDA code from the IR.


Filters
=============

.. automodule:: Backends.Cuda.Visitors.CM_Visitors
   :members: 


Templates
=============

Currently, templates are held inside :class:`Mutators` code.

Mutators
=============

A separate Mutator have been written for each OpenMP construct.
Their parent is :class:`Backends.Cuda.Mutators.Common`

.. automodule:: Backends.Cuda.Mutators.Common
   :members: AbstractCudaMutator, CudaTransformer


The following constructs have been implemented:

**OpenMP Parallel**

.. automodule:: Backends.Cuda.Mutators.CM_OmpParallel
   :members: 



**OpenMP Parallel For**

.. automodule:: Backends.Cuda.Mutators.CM_OmpParallelFor
   :members: 


**OpenMP for**

.. automodule:: Backends.Cuda.Mutators.CM_OmpFor
   :members: 



