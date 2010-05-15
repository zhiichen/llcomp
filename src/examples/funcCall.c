

void func (int a)
{
   int i;

   for (i = 0; i < 10; i++) 
      i = i + 1;
}

int main() {

   int x;
   int i;

  for (i = 0; i < 10; i++)
   {
	x += *func(i);
   }
}
