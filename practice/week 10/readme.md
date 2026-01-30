# **Практическая работа №10**

---

## Среда выполнения

```markdown
- GPU: NVIDIA GeForce RTX 3070 Ti (compute capability 8.6, 8 ГБ)  
- ОС: Windows 11  
- CUDA: 13.1  
- Драйвер: 591.74  
- Компиляция: `nvcc -arch=sm_86 -O2`
```

---

Структура файлов
```markdown
practice
      └── week 10
            │
            ├── t1.cpp (Задача 1)
            ├── t2.cu (Задача 2)
            ├── t3.cu (Задача 3)
            └── t4.cpp (Задача 4)
```

---

## Результаты компиляции
### Задача 1
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 10
$ g++ -std=c++17 -O2 -fopenmp t1.cpp -o t1

jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 10
$ ./t1
Number of threads: 12
Sequential part:   0.00100017 seconds
Parallel part:     0.00299978 seconds
Final sum:         5.00089e+06
```

### Задача 2
```markdown
C:\Users\jinx\Desktop\practices\week 10>nvcc -O3 t2.cu -o t2                                                                             
t2.cu
tmpxft_00001480_00000000-7_t2.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 10>t2.exe 
Coalesced version:    0.461 ms
Scattered version:    3.663 ms
```

### Задача 3
```markdown
C:\Users\jinx\Desktop\practices\week 10>nvcc -O3 t3.cu -o t3
t3.cu
tmpxft_00003188_00000000-7_t3.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 10>t3.exe
Asynchronous finished
```

### Задача 4
Количество процессов: 1
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 10
# "/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" -n 1 ./t4
Number of processes: 1
Execution time:      0.0069278 s
Computed sum:        1e+07
```

Количество процессов: 4
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 10
# "/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" -n 4 ./t4
Number of processes: 4
Execution time:      0.0032704 s
Computed sum:        1e+07
```
