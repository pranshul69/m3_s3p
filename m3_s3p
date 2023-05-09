#define CL_TARGET_OPENCL_VERSION 300
#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>

#define MAX 8
size_t global_size = MAX;
 size_t local_size = 1;



cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err;

int a[MAX], b[MAX], c[MAX];
cl_mem bufA, bufB, bufC;

cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }

    // Access a device
    // GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        // CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}

void init(int* vector);
void print_vector(int* vector);
void vector_addition(int* vectorA, int* vectorB, int* vectorC);
cl_device_id create_device();

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

void setup_openCL_device_context_queue_kernel();
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();

void init(int* vector) {
    for (int i = 0; i < MAX; i++) {
        vector[i] = rand() % 10;
    }
}

void print_vector(int* vector) {
    for (int i = 0; i < MAX; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

void vector_addition(int* vectorA, int* vectorB, int* vectorC) {
    for (int i = 0; i < MAX; i++) {
        vectorC[i] = vectorA[i] + vectorB[i];
    }
}



int main() {
    init(a);
    init(b);

    printf("Input vectors:\n");
    print_vector(a);
    print_vector(b);

    vector_addition(a, b, c);

    printf("Vector addition using CPU:\n");
    print_vector(c);

    setup_openCL_device_context_queue_kernel();
    setup_kernel_memory();
    copy_kernel_args();

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
    clWaitForEvents(1, &event);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, MAX * sizeof(int), c, 0, NULL, NULL);

    printf("Vector addition using OpenCL:\n");
    print_vector(c);

    free_memory();
}

void free_memory() {
    clReleaseKernel(kernel);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

void copy_kernel_args() {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufC);
    if (err < 0) {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory() {
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, MAX * sizeof(int), NULL, NULL);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, MAX * sizeof(int), NULL, NULL);
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, MAX * sizeof(int), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, MAX * sizeof(int), b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, MAX * sizeof(int), c, 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel() {
    device_id = create_device();
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, "matrix_mul.cl");

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    }

    kernel = clCreateKernel(program, "matrix+mul", &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    }
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    cl_program program;
    FILE* program_handle;
    char* program_buffer;
    size_t program_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &program_size);
        char* program_log = (char*)malloc(program_size + 1);
        program_log[program_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, program_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}
