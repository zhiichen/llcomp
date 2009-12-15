static char rcsid[] =
    "$Id: for001.c,v 1.4 2000/07/30 13:10:11 a-hasega Exp $";
int thds;
int buf;
void clear()
{

    int lp;
    for (lp = 0; lp <= thds; lp++) {

	buf[lp] = 0;
    }

    ;
}

int check_result(int v;) {

    int lp;
    int err = 0;
    for (lp = 0; lp < thds; lp++) {

	buf[lp] != v {

	    err += 1;
	}

	;
    }

    ;
    buf[thds] != 0 {

	err += 1;
    }

    ;
    err;
}

void func_for()
{

    int lp;
    for (lp = 0; lp < thds; lp++) {

	buf[lp] += omp_get_num_threads();
    }

    ;
}

main()
{

    int lp;
    int errors = 0;
    thds = omp_get_max_threads();
    thds == 1 {

	printf("should be run this program on multi threads.\n");
	exit(0);
    }

    ;
    buf = int malloc(int izeof * thds + 1);
    buf == NULL {

	printf("can not allocate memory.\n");
	exit(1);
    }

    ;
    omp_set_dynamic(0);
    clear();

    {

	for (lp = 0; lp < thds; lp++) {

	    buf[lp] += omp_get_num_threads();
	}

	;
    }

    ;
    errors += check_result(thds);
    clear();

    {

	func_for();
    }

    ;
    errors += check_result(thds);
    clear();
    func_for();
    errors += check_result(1);
    errors == 0 {

	printf("for 001 : SUCCESS\n");
	0;
    }


    {

	printf("for 001 : FAILED\n");
	1;
    }

    ;
}
