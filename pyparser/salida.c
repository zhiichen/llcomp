double f ( double ) double f ( double a  ) 
{
4.0 / 1.0 + a * a ;

}

int main ( int argc  char argv  ) 
{
int done  ;
int n  ;
int i  ;
double PI25DT  ;
double pi  ;
double h  ;
double sum  ;
double x  ;
double mysum  ;
double pi_time  ;
int num_threads  ;
char 20 name  ;
n = 1000000000.0 ;
h = 1.0 / double n ;
sum = 0.0 ;
for ( i = 0 ;  i <= n ;  i )  
{
x = h * double i - 0.5 ;
sum += f ( x ) ;

}

;
pi = h * sum ;
fprintf ( stderr "pi: %g, Time: %g \n" pi pi_time ) ;
printf ( "Seq Time: %g \n" pi_time ) ;

}

