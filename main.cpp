#include <iostream>
#include <cl/cl.h>
#include <vector>
#include <string>
#include <string.h>
#include <assert.h>
#include <SFML/System.hpp>
#include <algorithm>
#include <atomic>
#include <optional>
#include <array>

void get_platform_ids(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024] = {};
    cl_uint num_platforms;
    std::vector<cl_platform_id> clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;
    cl_uint i = 0;

    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);

    if(ciErrNum != CL_SUCCESS)
    {
        throw std::runtime_error("Bad clGetPlatformIDs call " + std::to_string(ciErrNum));
    }
    else
    {
        if(num_platforms == 0)
        {
            throw std::runtime_error("No available platforms");
        }
        else
        {
            clPlatformIDs.resize(num_platforms);

            ciErrNum = clGetPlatformIDs(num_platforms, &clPlatformIDs[0], NULL);

            for(i = 0; i < num_platforms; i++)
            {
                ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);

                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL || strstr(chBuffer, "AMD") != NULL)// || strstr(chBuffer, "Intel") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];

                        //printf("Picked Platform: %s\n", chBuffer);
                    }
                }
            }

            if(*clSelectedPlatformID == NULL)
            {
                *clSelectedPlatformID = clPlatformIDs[num_platforms-1];
            }
        }
    }
}

#define CHECK(x) do{if(auto err = x; err != CL_SUCCESS) {throw std::runtime_error("Got error " + std::to_string(err));}}while(0)

std::string program_source =
R"(
bool iswspace(char in)
{
   return in == ' ' || in == '\f' || in == '\n' || in == '\r' || in == '\t' || in == '\v';
}

__kernel
void some_func(__global char* data, __global int* words, __global int* newlines, __global size_t* real_size)
{
    size_t id = get_global_id(0);

    if(id >= *real_size)
        return;

    char mdata = data[id];

    if(mdata == '\n')
        atomic_inc(newlines);

    if(id == 0)
        return;

    bool m1 = iswspace(data[id]);
    bool m2 = !iswspace(data[id - 1]);

    if(m1 && m2)
        atomic_inc(words);
}
)";

cl_event exec_1d(cl_command_queue cqueue, cl_kernel kernel, const std::vector<cl_mem>& args, size_t global_ws, size_t local_ws, const std::vector<cl_event>& waitlist)
{
    for(int i=0; i < (int)args.size(); i++)
    {
        clSetKernelArg(kernel, i, sizeof(cl_mem), &args[i]);
    }

    if(local_ws != 0)
    {
        if((global_ws % local_ws) != 0)
        {
            int rem = (global_ws % local_ws);

            global_ws -= rem;
            global_ws += local_ws;
        }
    }

    cl_event evt;

    if(waitlist.size() == 0)
    {
        CHECK(clEnqueueNDRangeKernel(cqueue, kernel, 1, nullptr, &global_ws, &local_ws, 0, nullptr, &evt));
    }
    else
    {
        CHECK(clEnqueueNDRangeKernel(cqueue, kernel, 1, nullptr, &global_ws, &local_ws, waitlist.size(), &waitlist[0], &evt));
    }

    return evt;
}

struct buffer
{
    cl_mem mem;
    size_t allocation = 0;

    void alloc(cl_context ctx, size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
        mem = clCreateBuffer(ctx, flags, bytes, nullptr, nullptr);
        allocation = bytes;
    }

    cl_event async_write(cl_command_queue cqueue, void* ptr, size_t bytes)
    {
        assert(bytes <= allocation);

        cl_event event;

        CHECK(clEnqueueWriteBuffer(cqueue, mem, CL_FALSE, 0, bytes, ptr, 0, nullptr, &event));

        return event;
    }

    cl_event async_write(cl_command_queue cqueue, void* ptr, size_t bytes, const std::vector<cl_event>& events)
    {
        assert(bytes <= allocation);

        cl_event event;

        if(events.size() == 0)
            CHECK(clEnqueueWriteBuffer(cqueue, mem, CL_FALSE, 0, bytes, ptr, 0, nullptr, &event));
        else
            CHECK(clEnqueueWriteBuffer(cqueue, mem, CL_FALSE, 0, bytes, ptr, events.size(), &events[0], &event));

        return event;
    }

    cl_event async_read(cl_command_queue cqueue, void* ptr, size_t bytes, const std::vector<cl_event>& events)
    {
        cl_event event;

        const cl_event* fptr = events.size() > 0 ? &events[0] : nullptr;

        CHECK(clEnqueueReadBuffer(cqueue, mem, CL_FALSE, 0, bytes, ptr, events.size(), fptr, &event));

        return event;
    }
};

struct pcie_mem
{
     cl_mem clptr;
     size_t pcie_size = 0;

     void allocate(cl_context ctx, size_t bytes)
     {
        clptr = clCreateBuffer(ctx, CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, nullptr);
        pcie_size = bytes;
     }

     std::pair<void*, cl_event>
     map(cl_command_queue cqueue, cl_map_flags flags)
     {
        cl_event evt;
        void* ptr = clEnqueueMapBuffer(cqueue, clptr, CL_FALSE, flags, 0, pcie_size, 0, nullptr, &evt, nullptr);

        return {ptr, evt};
     }

     void unmap(cl_command_queue cqueue, void* ptr)
     {
         clEnqueueUnmapMemObject(cqueue, clptr, ptr, 0, nullptr, nullptr);
     }
};

void wait(cl_event evt)
{
    clWaitForEvents(1, &evt);
}

size_t file_size(const std::string& file)
{
    FILE *f = fopen(file.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fclose(f);

    return fsize;
}

void read_file_into(FILE* pFile, size_t file_size, void* ptr)
{
    fread(ptr, file_size, 1, pFile);
}

template<typename T>
struct double_buffered
{
    #define FLIPSIZE 4

    int which = 0;
    std::array<T, FLIPSIZE> data = {};

    T& get(int offset)
    {
        return data[(which + offset) % FLIPSIZE];
    }

    void flip()
    {
        which++;
        which %= FLIPSIZE;
    }
};

int main()
{
    cl_platform_id pid = {};
    get_platform_ids(&pid);

    cl_uint num_devices = 0;
    cl_device_id devices[100] = {};

    CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, devices, &num_devices));

    cl_device_id selected_device = devices[0];

    cl_context_properties props[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)pid,
        0
    };

    cl_int error = 0;

    cl_context ctx = clCreateContext(props, 1, &selected_device, nullptr, nullptr, &error);

    if(error != CL_SUCCESS)
        throw std::runtime_error("Failed to create context " + std::to_string(error));

    const char* cprogram_source = program_source.c_str();
    size_t cprogram_len = program_source.size();

    cl_program program = clCreateProgramWithSource(ctx, 1, &cprogram_source, &cprogram_len, nullptr);

    std::string build_options = "-cl-fast-relaxed-math";

    cl_int build_status = clBuildProgram(program, 1, &selected_device, build_options.c_str(), nullptr, nullptr);

    if(build_status != CL_SUCCESS)
    {
        printf("Build error\n");

        cl_build_status bstatus;
        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &bstatus, nullptr);

        printf("Err: %i\n", bstatus);

        assert(bstatus == CL_BUILD_ERROR);

        std::string log;
        size_t log_size;

        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        log.resize(log_size + 1);

        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], nullptr);

        printf("%s\n", log.c_str());

        return 4;
    }

    cl_kernel kernel = clCreateKernel(program, "some_func", &error);

    if(error != CL_SUCCESS)
        throw std::runtime_error("Could not create kernel " + std::to_string(error));

    cl_command_queue cqueue = clCreateCommandQueue(ctx,  selected_device, 0, nullptr);
    cl_command_queue async_queue = clCreateCommandQueue(ctx, selected_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, nullptr);

    sf::Clock init;

    ///would be very possible to load the file in chunks, and overlap gpu data transfer and execution with file loading
    double_buffered<pcie_mem> file;
    double_buffered<void*> fptrs;
    double_buffered<buffer> gpu_data;

    size_t file_to_read_size = file_size("test.txt");

    #define CHUNK_SIZE 1024 * 1024 * (16 / FLIPSIZE)

    cl_event last_event;

    for(int i=0; i < FLIPSIZE; i++)
    {
        file.get(i).allocate(ctx, CHUNK_SIZE);

        gpu_data.get(i).alloc(ctx, CHUNK_SIZE, CL_MEM_READ_ONLY);

        auto [cptr1, file_event1] = file.get(i).map(cqueue, CL_MAP_READ | CL_MAP_WRITE);

        fptrs.data[i] = cptr1;

        last_event = file_event1;
    }

    wait(last_event);

    FILE* pFile = fopen("test.txt", "rb");

    int zero = 0;

    buffer word_count;
    word_count.alloc(ctx, 4);
    cl_event evt2 = word_count.async_write(cqueue, &zero, sizeof(zero));

    buffer newline_count;
    newline_count.alloc(ctx, 4);
    cl_event evt3 = newline_count.async_write(cqueue, &zero, sizeof(zero));

    //wait(evt2);
    //wait(evt3);

    double_buffered<std::optional<cl_event>> unfinished_events;

    int remaining = file_to_read_size;

    while(remaining > 0)
    {
        int to_read = std::min(remaining, CHUNK_SIZE);

        int* ntoread = new int(to_read);

        if(unfinished_events.get(0) != std::nullopt)
            wait(*unfinished_events.get(0));

        read_file_into(pFile, to_read, fptrs.get(0));

        cl_event gpudatawrite = gpu_data.get(0).async_write(async_queue, fptrs.get(0), to_read);

        buffer real_size;
        real_size.alloc(ctx, 8);
        cl_event evt4 = real_size.async_write(cqueue, ntoread, sizeof(to_read));

        cl_event kevent = exec_1d(cqueue, kernel, {gpu_data.get(0).mem, word_count.mem, newline_count.mem, real_size.mem}, to_read, 64, {gpudatawrite, evt4});
        unfinished_events.get(0) = kevent;

        gpu_data.flip();
        fptrs.flip();
        unfinished_events.flip();

        remaining -= to_read;
    }

    fclose(pFile);

    int cword_count = 0;
    int cnewline_count = 0;

    word_count.async_read(cqueue, &cword_count, sizeof(int), {});
    wait(newline_count.async_read(cqueue, &cnewline_count, sizeof(int), {}));

    printf("WORD COUNT %i ", cword_count);
    printf("NEWLINE COUNT %i ", cnewline_count);

    printf("Time elapsed in s %lf\n", init.getElapsedTime().asMicroseconds() / 1000. / 1000.);

    return 0;
}
