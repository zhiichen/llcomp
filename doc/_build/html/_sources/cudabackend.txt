CUDA Backend
*****************
.. module:: CudaBackend 
   :synopsis: Converts IR to CUDA


The :mod:`CudaBackend` module contains a set of Mutators, Filters and Templates
which creates CUDA code from the IR.


Filters
=============

.. automodule:: Backends.CudaBackend.Visitors.CM_Visitors
   :members: 


Templates
=============

Currently, templates are held inside :class:`Mutators` code.

Mutators
=============

A separate Mutator have been written for each OpenMP construct.
Their parent is :class:`Backends.CudaBackend.Mutators.Common`

.. automodule:: Backends.CudaBackend.Mutators.Common
   :members: AbstractCudaMutator


The following constructs have been implemented:

**OpenMP Parallel**

.. automodule:: Backends.CudaBackend.Mutators.CM_OmpParallel
   :members: 



**OpenMP Parallel For**

.. automodule:: Backends.CudaBackend.Mutators.CM_OmpParallelFor
   :members: 


**OpenMP for**

automodule:: Backends.CudaBackend.Mutators.CM_OmpFor
members: 



