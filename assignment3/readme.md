# **Assignment 3**
**Архитектура GPU и оптимизация CUDA-программ**

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
assignment3
│
├── t1.cu (Задача 1)
├── t2.cu (Задача 2)
├── t3.cu (Задача 3)
└── t4.cu (Задача 4)
```

---

## Результаты компиляции
### Задача 1
```markdown
C:\Users\jinx\Desktop\assignment3>t1.exe
Global memory version : 0.092 ms
Shared memory version : 0.013 ms
```

### Задача 2
```markdown
C:\Users\jinx\Desktop\assignment3>t2.exe
Block size      Time (ms)
------------------------
128             0.071008
256             0.052224
512             0.070656
```

### Задача 3
```markdown
C:\Users\jinx\Desktop\assignment3>t3.exe 
Coalesced   access : 0.063 ms
Uncoalesced access : 0.072 ms
```

### Задача 4
```markdown
C:\Users\jinx\Desktop\assignment3>t4.exe

  block   |   grid   |   avg time (ms)
------------------------------------------
     64  |   15625  |       0.025 ms
    128  |    7813  |       0.027 ms
    192  |    5209  |       0.027 ms
    256  |    3907  |       0.027 ms
    384  |    2605  |       0.026 ms
    512  |    1954  |       0.027 ms
    768  |    1303  |       0.026 ms
   1024  |     977  |       0.026 ms

Results:
Best: block = 64, time = 0.025 ms
Worst: block = 256, time = 0.027 ms
```
