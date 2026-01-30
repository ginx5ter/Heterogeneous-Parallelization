__kernel void vector_add(
    __global const float* inputA,
    __global const float* inputB,
    __global float* outputC)
{
    const int index = get_global_id(0);
    outputC[index] = inputA[index] + inputB[index];
}