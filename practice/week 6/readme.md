# **Практическая работа №6**

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
      └── week 6
            ├── task 1
            │      ├── kernel_vector_add.cl
            │      └── t1.cpp (Задача 1)
            └── task 2
                   ├── kernel_matrix_mul.cl
                   └── t1.cpp (Задача 1)
```  

---

## Результаты компиляции
### Задача 1
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 6/task 1
$ ./vecadd
Vector addition time: 0.0024303 s
Result[0] = 0, Result[last] = 749999
```

### Задача 2
```markdown
jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 6/task 2
$ g++ -O2 -std=c++17 -Wno-deprecated-declarations t1.cpp -lOpenCL -o vecadd
In file included from C:/msys64/mingw64/include/CL/cl.h:20,
                 from t1.cpp:1:
C:/msys64/mingw64/include/CL/cl_version.h:22:104: note: '#pragma message: cl_version.h: CL_TARGET_OP
ENCL_VERSION is not defined. Defaulting to 300 (OpenCL 3.0)'
   22 | #pragma message("cl_version.h: CL_TARGET_OPENCL_VERSION is not defined. Defaulting to 300 (O
penCL 3.0)")
      |
           ^

jinx@DESKTOP-RL3A54D MINGW64 /c/Users/jinx/Desktop/practices/week 6/task 2
$ ./vecadd
Matrix multiplication correct: YES
```
