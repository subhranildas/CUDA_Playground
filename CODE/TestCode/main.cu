#include <iostream>
using namespace std;

int main()
{
    cout << "Hello World" << endl;
    int x = 10;
    int *ptr = &x;
    cout << "Address of x:" << ptr << endl;
    cout << "Value of x:" << *ptr << endl;

    return 0;
}