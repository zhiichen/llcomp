Backends
*****************
.. module:: Backends
   :synopsis: Repository of implemented backends

The :mod:`Backends` module packages the different backends implemented on the compiler.


The :mod:`Common` module contains common classes for all backends.

The :mod:`DotBackend` module is able to translate the :term:`Internal
Representation` (IR) to :term:`Dot language`, which may be printed with
graphviz.

The :mod:`C` module contains writers capable of converting the IR to
C or OpenMP code.

Module :mod:`Cuda` encapsulates Mutators, Visitors and Writers, capable of
translating the IR to CUDA code.

.. toctree::
   :maxdepth: 2
   
   common.rst
   dotbackend.rst
   cbackend.rst
   cudabackend.rst

