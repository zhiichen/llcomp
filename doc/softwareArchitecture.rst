Software Architecture
**********************************

.. _layered_design:
.. figure:: images/layered_design.png
   :align:  center

In the diagram (layered_design_), the different layers of the framework are exposed. 

The uppermost level contains the :mod:`Frontend`, which gives the tools required to transform 
the source code into the internal representation.

The :mod:`MiddleEnd` module encapsulates transformations from the IR to the IR, for example, 
loop optimizations or type data conversions.

Finally :mod:`Backends` module contains all the implemented backends

Tools to manipulate the internal representation (and do some other stuff), are packaged 
on the :mod:`Tools` module.

In addition, some utils and examples are presented in order to show the capabilities of the framework.


Architecture details
********************************

.. toctree::
   :maxdepth: 2

   frontend.rst
   backends.rst
   tools.rst

