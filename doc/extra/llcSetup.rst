|llCoMP| Setup
==================================


|llCoMP| Requirements
**********************************


Previously of installing |llCoMP| on your system, you must satisfy the following requirements:

1. Bison and Flex
2. python-ply
3. python-yaml
   Only required if you want to rebuild the frontend
4. pycparser
   Clone it from the pycparser project source page:

   hg clone https://pycparser.googlecode.com/hg/ pycparser

   Check that works properly by running the z_test.py script


|llCoMP| Install
**********************************


1. Clone the source code repository
    a. If you want the last branch, clone from google code:

	hg clone http://llcomp.googlecode.com/hg/ llcomp

    b. If you want the heavy work-in-progress branch, get the code from the private repository
	at bitbucket. Note that you need a valid account.


2. Move to the destination dir and create a config_local.py file (you can copy the config.py file as an example): :: 
   
	   # Local config file for llcomp
	   # Current work dir
	   WORKDIR="/home/user/llcomp" # llcomp setup dir
	   # Location of cuda files
	   CUDA_INSTALL_DIR="/usr/local/cuda/"
	   # C includes for templates
	   # pycparser dir
	   PYCPARSER_DIR="/home/user/pycparser/"


3. In order to use the |llc| syntax, it is necessary to patch pycparser. Patch files are located in the utils/pycparser dir.

   a. Copy all files under the utils/pycparser dir to the directory where you previously installed pycparser
   b. Run the update_patch script with the apply argument (bash update_patch.sh apply)
   c. Run the new z_test.py to check that new syntax has been correctly installed.

4. Export the llcomp and pycparser dir to the PYTHONPATH ::

    export PYTHONPATH='/home/user/llcomp':'/home/user/pycparser'

    
5. Run checkall script to run tests and check for a correct installation ::
    
     python tests/checkall.py 


Known Problems
**********************************

1. In order to fasten compilation, |llCoMP| uses a cached version of parsing tables. This caching should be disabled if any change have been made to the frontend. Otherwise, the run will fail. To disable cache, in the module :mod:`Frontend.Parse` the function parse_source should look like this. If you want to re-enable cache, set **yacc_optimize** and **lex_optimize** to **True**  ::


     ast = c_parser.CParser(lex_optimize = False, yacc_optimize = False).parse(
     stripped_source, filename = code_name)




