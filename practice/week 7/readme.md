# **Практическая работа №7**

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
      └── week 7
            │
            ├── t1.cu (Задача 1)
            ├── t2.cu (Задача 2)
            └── t3.cu (Задача 3)
```

---

## Результаты компиляции
### Задача 1
```markdown
C:\Users\jinx\Desktop\practices\week 7>nvcc -O3 t1.cu -o t1
t1.cu
tmpxft_000008c4_00000000-7_t1.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 7>t1.exe
Size of array:      1048576
Sum (CPU)  = 1.04845e+06
Sum (GPU)  = 1.04841e+06
Difference      = 43.1875
```

### Задача 2
```markdown
C:\Users\jinx\Desktop\practices\week 7>nvcc -O3 t2.cu -o t2 
t2.cu
tmpxft_00001e3c_00000000-7_t2.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 7>t2.exe 
Last element: 1024  (expected 1024)
```

### Задача 3
```markdown
C:\Users\jinx\Desktop\practices\week 7>nvcc -O3 t3.cu -o t3
t3.cu
tmpxft_00001d60_00000000-7_t3.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 7>t3.exe
GPU:  0.054272 ms
CPU:  0.0027 ms
```
