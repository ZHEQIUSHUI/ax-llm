#include "runner/ax_model_runner/ax_parallel_runner.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include <axcl_rt_memory.h>
#include <axcl.h>
#include "cmdline.hpp"
#include "timer.hpp"
#include <axcl_rt_p2p.h>
#include <fstream>

bool write_file(const std::string &filename, const void *data, size_t size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    file.write(static_cast<const char *>(data), size);
    if (!file.good())
    {
        std::cerr << "Error: Could not write to file " << filename << std::endl;
    }
    file.close();
    return true;
}

std::vector<char> read_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        std::cerr << "Error: Could not read file " << filename << std::endl;
        return {};
    }
    file.close();
    return buffer;
}

int main(int argc, char **argv)
{
    int DEFAULT_LOOP_COUNT = 10;
    cmdline::parser parser;
    parser.add<int>("loop_count", 'l', "loop count", false, DEFAULT_LOOP_COUNT);
    parser.parse_check(argc, argv);
    std::vector<int> devices = {
        0,
    };
    axclInit(0);

    for (auto &devid : devices)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return false;
        }
    }

    DEFAULT_LOOP_COUNT = parser.get<int>("loop_count");

    ax_runner_ax650 runner;
    if (runner.init("axmodel_qwen2.5_1.5b_chunked_nogptq_bf16/qwen2_post.axmodel", 0) != 0)
    {
        std::cout << "init model error" << std::endl;
        return -1;
    }

    std::vector<char> buffer = read_file("/home/axera/ax-llm/build/post_input.bin");

    if (buffer.size() != runner.get_input(0).nSize)
    {
        printf("input.bin size is not equal to model input size, model input size=%d, input.bin size=%ld\n", runner.get_input(0).nSize, buffer.size());
        return -1;
        /* code */
    }

    axcl_Memcpy((void *)runner.get_input(0).phyAddr, buffer.data(), buffer.size(), AXCL_MEMCPY_HOST_TO_DEVICE, runner.get_devid());

    timer t_cost;
    for (size_t i = 0; i < DEFAULT_LOOP_COUNT; i++)
    {
        t_cost.start();
        runner.inference();
        printf("inference cost: %.2f ms\n", t_cost.cost());
        fflush(stdout);
    }

    std::vector<char> buffer_out = read_file("/home/axera/ax-llm/build/post_output.bin");
    if (buffer_out.size() != runner.get_output(0).nSize)
    {
        printf("output.bin size is not equal to model output size, model output size=%d, output.bin size=%ld\n", runner.get_output(0).nSize, buffer_out.size());
        return -1;
    }

    axcl_Memcpy(runner.get_output(0).pVirAddr, (void *)runner.get_output(0).phyAddr, runner.get_output(0).nSize, AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid());

    auto ret = memcmp(buffer_out.data(), runner.get_output(0).pVirAddr, buffer_out.size());
    if (ret != 0)
    {
        printf("output.bin is not equal to model output\n");
    }

    runner.deinit();

    for (auto &devid : devices)
        axcl_Exit(devid);

    axclFinalize();

    return 0;
}