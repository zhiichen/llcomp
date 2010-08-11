Future Work
*****************

Work in progress concerning this topic includes the following:


* To increase the number of algorithms parallelized using our compiler, with particular attention to commercial applications
* To study and implement additional compiler optimizations that will enhance the performance of the target code
* To extend the \llc{} syntax to capture additional information from the programmer for better adaption of the translation to the target architecture
* To study the generation of hybrid CUDA+OpenMP code

Some future work may be done within the CUDA backend.

* To improve locality through a better use of the memory hierarchy.
* To use the texture memory to store read-only data.
* An intelligent balance of load between host and device.
