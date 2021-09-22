#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

/* Also see: https://www.cnblogs.com/CCBB/archive/2010/01/15/1648827.html */
int compare(const void *a, const void *b)
{
    double pa =* ((double*)a);
    double pb =* ((double*)b);
    return (pa > pb) ? 1 : -1;
}

int main(void)
{
	double a[10] = {5.6, 6, 4, 3, 7.58, 0 ,8, 9, 2, 1};
   qsort(a, 10, sizeof(double), compare);
   for (int i = 0; i < 10; i++)
		cout << a[i] << " " << endl;
	
	cout << "Length of float:" << sizeof(double) << endl;
}
