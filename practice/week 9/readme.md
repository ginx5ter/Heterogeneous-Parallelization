# **Практическая работа №9**

---

## Среда выполнения

```markdown
- GPU: NVIDIA GeForce RTX 3070 Ti (compute capability 8.6, 8 ГБ)
- CPU: AMD Ryzen 5 5600X  
- ОС: Windows 10  
- CUDA: 13.1  
- Драйвер: 591.74  
- Компиляция: `nvcc -arch=sm_86 -O2`
```

---

Структура файлов
```markdown
practice
      └── week 9
            │
            ├── t1.cpp (Задача 1)
            ├── t2.cpp (Задача 2)
            └── t3.cpp (Задача 3)
```

---

## Результаты компиляции
### Задача 1
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 9
# mpic++ -std=c++17 -O2 -fopenmp t1.cpp -o t1

jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 9
# ./t1
Elements: 1000000    Processes: 1
Mean:     0.500456
Std dev:  0.288574
Time:     0.0101343 s
```

### Задача 2
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 9
# mpic++ -std=c++17 -O2 -fopenmp t2.cpp -o t2

jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 9
# ./t2
Size: 512    Processes: 1
Time: 0.018523 s
First 8 solution components:
  x[0] = 0.00149652
  x[1] = 0.00330883
  x[2] = -0.000342359
  x[3] = -0.00135555
  x[4] = -0.00111862
  x[5] = 0.000531589
  x[6] = 0.000175349
  x[7] = -0.00114371
```

### Задача 3
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 9
# "/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" -n 1 ./t3 256
Nodes: 256    Processes: 1
Time:  0.0261729 s
Top-left 10x10 distances:
   0    7    4    6    7    7    7    2    7    6
   4    0    4    4    2    5    5    2    5    5
   7    3    0    5    5    5    5    5    6    6
   4    4    4    0    5    4    5    5    5    4
   5    5    3    4    0    5    3    4    3    4
   3    5    5    5    5    0    5    4    5    6
   5    4    4    4    5    2    0    5    4    4
   4    5    2    4    5    5    6    0    6    4
   2    5    2    5    6    6    4    4    0    3
   7    7    6    6    5    6    7    4    5    0
```
