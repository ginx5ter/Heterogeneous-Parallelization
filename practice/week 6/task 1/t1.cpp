#include <CL/cl.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#define CL_CHECK(err)                                                          \
  do {                                                                         \
    if (err != CL_SUCCESS) {                                                   \
      std::cerr << "OpenCL error code: " << err << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

std::string read_kernel_source(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file) {
    std::cerr << "Cannot open kernel file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

int main() {
  constexpr size_t ELEMENT_COUNT = 1'000'000;
  constexpr size_t BYTE_SIZE = ELEMENT_COUNT * sizeof(float);

  std::vector<float> vecA(ELEMENT_COUNT), vecB(ELEMENT_COUNT),
      vecResult(ELEMENT_COUNT);

  for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
    vecA[i] = static_cast<float>(i) * 0.5f;
    vecB[i] = static_cast<float>(i) * 0.25f;
  }

  cl_int error;
  cl_platform_id platform_id = nullptr;
  cl_device_id device_id = nullptr;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, nullptr));
  CL_CHECK(
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, nullptr));

  cl_context ctx =
      clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
  CL_CHECK(error);

  cl_command_queue command_queue =
      clCreateCommandQueue(ctx, device_id, 0, &error);
  CL_CHECK(error);

  std::string kernel_code = read_kernel_source("kernel_vector_add.cl");
  const char *kernel_source_ptr = kernel_code.c_str();
  size_t kernel_source_length = kernel_code.length();

  cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_source_ptr,
                                                 &kernel_source_length, &error);
  CL_CHECK(error);

  error = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    size_t log_length = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_length);
    std::vector<char> build_log(log_length);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_length,
                          build_log.data(), nullptr);
    std::cerr << "Build error log:\n" << build_log.data() << std::endl;
    exit(EXIT_FAILURE);
  }

  cl_kernel kernel = clCreateKernel(program, "vector_add", &error);
  CL_CHECK(error);

  cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  BYTE_SIZE, vecA.data(), &error);
  CL_CHECK(error);
  cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  BYTE_SIZE, vecB.data(), &error);
  CL_CHECK(error);
  cl_mem bufferC =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, BYTE_SIZE, nullptr, &error);
  CL_CHECK(error);

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC));

  size_t global_work_size = ELEMENT_COUNT;

  auto t_start = std::chrono::steady_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr,
                                  &global_work_size, nullptr, 0, nullptr,
                                  nullptr));
  CL_CHECK(clFinish(command_queue));
  auto t_end = std::chrono::steady_clock::now();

  CL_CHECK(clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, BYTE_SIZE,
                               vecResult.data(), 0, nullptr, nullptr));

  std::chrono::duration<double> duration = t_end - t_start;
  std::cout << "Vector addition time: " << duration.count() << " s\n";
  std::cout << "Result[0] = " << vecResult[0]
            << ", Result[last] = " << vecResult[ELEMENT_COUNT - 1] << "\n";

  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(ctx);

  return 0;
}