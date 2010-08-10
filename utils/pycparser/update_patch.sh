
if [ "$#" -ne 1 ];then
  echo " Incorrect option. Use \"apply\" or \"update\""
  exit 
fi

APPLY=$1

FILES="_ast_gen.py _c_ast.yaml c_lexer.py c_parser.py"
for file in $FILES; do
	PATCHFILE=`echo -n "$file" | cut -f 1 -d . | tr -d "\n" ; echo ".patch"`
        if [ "$APPLY" = "apply" ]; then
		patch -p 0 pycparser/$file < $PATCHFILE
        else 
		hg diff pycparser/$file > $PATCHFILE
	fi
        echo " Updating $file"
done

if [ "$APPLY" != "apply" ]; then
	cp *.patch ../llcomp/utils/pycparser/
	cp z_test.py ../llcomp/utils/pycparser/
fi
