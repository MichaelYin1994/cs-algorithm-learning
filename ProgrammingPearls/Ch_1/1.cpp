#include<iostream>
#include<set>
#include <stdlib.h>
using namespace std;

int main(void)
{
	set<int> S;
	int i;
	set<int>::iterator j;
	cout<<"Please input int numbers: "<<endl;
	while(cin>>i)
		S.insert(i);
	for (j=S.begin(); j!=S.end(); j++)
		cout<<*j<<"\n";
	
	return 0;
}
