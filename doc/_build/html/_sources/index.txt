.. llCoMP documentation master file, created by
   sphinx-quickstart on Mon Jun 21 11:41:53 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|llCoMP| Developer Documentation
========================================================================

.. sectionauthor:: Ruymán Reyes Castro <rreyes@ull.es>

|llCoMP| is a translator framework designed for *fast prototyping*. 
With a small development effort, developers can build translators from OpenMP/C to different High
Performance Computing languages, libraries and frameworks. Currently we have
implemented the CUDA Backend, but we have plans to integrate more backends (see :doc:`futureWork`).

|llCoMP| MiddleEnd capabilities allows users to apply a predefened set of high level
optimization techiques (like loop unrolling or loop flatening), or to implement their own
set of high level optimizations.


|llc| Language
**********************************

|llc| language has been presented on several papers [Dorta:2006:BSL]_, as an effective language
for high performance computing. |llc| is a language based on |OpenMP|/C where parallelism 
is expressed using compiler directives. 

See more about |llc| in :doc:`llcLanguage`.



|llCoMP| Setup
**********************************

In order to setup inst


|llCoMP| Additional documentation and HOWTOs
**************************************************

Some tutorials and HOWTO about |llCoMP|.


.. toctree::
   :glob:

   extra/*

.. comment
  * :doc:`Install |llCoMP| in your system <extra/llcSetup.rst>`.
  * Tutorial: Convert a file from C to CUDA
  * Tutorial: Write a simple mutator for loop interchange
  * Tutorial: Add a new language keyword
  * HOWTO use DotDebugTool

Contents
********************************

.. toctree::
   :maxdepth: 2

   softwareArchitecture.rst
   glossary.rst
   futureWork.rst
   publications.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

