# **Assignment 4**
**Гибридные и распределённые параллельные вычисления**

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
assignment3
│
├── t1.cu (Задача 1)
├── t2.cu (Задача 2)
├── t3.cpp (Задача 3)
└── t4.cpp (Задача 4)
```

---

## Результаты компиляции
### Задача 1
```markdown
C:\Users\jinx\Desktop\assignment4>t1.exe
Atomic reduction (N = 100000)
CPU result = 100000.0   time = 0.065 ms
GPU result = 100000.0   time = 0.241 ms
```

### Задача 2
```markdown
C:\Users\jinx\Desktop\assignment4>t2.exe
Prefix sum (inclusive, shared memory, block size 1024)
CPU time: 0.0007 ms
GPU time: 0.0563 ms
```

### Задача 3
```markdown
C:\Users\jinx\Desktop\assignment4>t3.exe
Hybrid CPU+GPU
Array size  : 1000000 elements
CPU only    : 0.146 ms
GPU only    : 0.094 ms
Hybrid      : 0.464 ms
```

### Задача 4
Количество процессов: 1
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/assignment4
$ mpiexec -n 1 ./t4.exe
Distributed sum using MPI
Number of processes : 1
Total elements      : 1000000
Global sum          : 1000000.0
Time                : 0.0008 seconds
```

Количество процессов: 4
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/assignment4
$ mpiexec -n 4 ./t4.exe
Distributed sum using MPI
Number of processes : 4
Total elements      : 1000000
Global sum          : 1000000.0
Time                : 0.0004 seconds
```
