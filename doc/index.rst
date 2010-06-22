.. llCoMP documentation master file, created by
   sphinx-quickstart on Mon Jun 21 11:41:53 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to llCoMP's documentation
==================================

.. sectionauthor:: Ruymán Reyes Castro <rreyes@ull.es>

llCoMP is a translator framework designed for *fast prototyping*. 
With little effort, you can build translators from OpenMP/C to different High
Performance Computing languages, libraries and frameworks.  Currently we have
implemented the CUDA Backend, but we have plans to implement new ones.


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


Contents:

.. toctree::
   :maxdepth: 2

   frontend.rst
   backends.rst
   tools.rst
   glossary.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

