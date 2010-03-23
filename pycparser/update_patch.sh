
FILES="_ast_gen.py _c_ast.yaml c_lexer.py c_parser.py"
for file in $FILES; do
	PATCHFILE=`echo -n "$file" | cut -f 1 -d . | tr -d "\n" ; echo ".patch"`
	svn diff pycparser/$file > $PATCHFILE
        echo " Updating $file"
done

cp *.patch ../llcomp/pycparser/
cp z_test.py ../llcomp/pycparser/

