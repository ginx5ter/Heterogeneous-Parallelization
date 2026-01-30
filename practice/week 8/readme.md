# **Практическая работа №8**

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
            └── t1.cu (Задача 1)
```

---

## Результаты компиляции
### Задача 1
```markdown
C:\Users\jinx\Desktop\practices\week 8>nvcc -O3 t1.cu -o t1
t1.cu
tmpxft_00003a24_00000000-7_t1.cudafe1.cpp

C:\Users\jinx\Desktop\practices\week 8>t1.exe
Hybrid processing: CPU vs GPU vs Hybrid

Size of the array:       1000000
Time (only CPU):   0.1057 ms
Time (only GPU):   0.092768 ms
Time (hybrid):       0.5831 ms
```
