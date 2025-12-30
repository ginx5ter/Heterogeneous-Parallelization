#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const int size = 50000;
    int *array = new int[size];

    for (int i = 0; i < size; ++i)
    {
        array[i] = std::rand() % 100 + 1;
    }

    long long sum = 0;
    for (int i = 0; i < size; ++i)
    {
        sum += array[i];
    }
    double average = static_cast<double>(sum) / size;

    std::cout << "avg is: " << average << std::endl;

    delete[] array;

    return 0;
}