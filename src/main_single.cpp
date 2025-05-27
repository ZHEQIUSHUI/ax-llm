#include "runner/ax_model_runner/ax_parallel_runner.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include <axcl_rt_memory.h>
#include <axcl.h>
#include "cmdline.hpp"
#include "bfloat16.hpp"
#include "timer.hpp"
#include <axcl_rt_p2p.h>
#include <fstream>
#include <complex>

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

float consine_similarity(const float *a, const float *b, int size)
{
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        dot_product += a[i] * b[i];
    }
    for (int i = 0; i < size; ++i)
    {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return dot_product / (norm_a * norm_b);
}

int main(int argc, char **argv)
{
    int DEFAULT_LOOP_COUNT = 10;
    cmdline::parser parser;
    parser.add<int>("loop_count", 'l', "loop count", false, DEFAULT_LOOP_COUNT);
    parser.parse_check(argc, argv);
    std::vector<int> devices = {0, 1, 2, 3};
    axclInit(0);

    for (auto &devid : devices)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return false;
        }
    }

    AXCL_P2P_UNIT_HANDLE p2p_handle = nullptr;
    {
        axclrtDeviceList device_list;
        if (const axclError ret = axclrtGetDeviceList(&device_list); AXCL_SUCC != ret || 0 == device_list.num)
        {
            printf("[ERROR] no device is connected.\n");
        }
        printf("[INFO] device num: %d.\n", device_list.num);
        if (2 > device_list.num)
        {
            printf("[ERROR] device num is less than 2.\n");
            return 2;
        }

        size_t p2p_cmm_size = 8 * 1024 * 1024;
        axclrtP2PUnitInfo p2p_unit;
        p2p_unit.u32DeviceNum = devices.size();
        for (uint32_t i = 0; i < p2p_unit.u32DeviceNum; ++i)
        {
            p2p_unit.n32DeviceId[i] = device_list.devices[devices[i]];
            p2p_unit.u32DeviceMemSize[i] = p2p_cmm_size;
        }

        if (const auto ret = axclrtCreateP2PUnit(&p2p_unit, &p2p_handle); AXCL_SUCC != ret)
        {
            printf("[ERROR] axcl init p2p unit fail, ret = 0x%x\n", ret);
            return -1;
        }
        else
        {
            std::cout << "[INFO] p2p unit created." << std::endl;
        }
    }

    DEFAULT_LOOP_COUNT = parser.get<int>("loop_count");

    // std::vector<ax_parallel_runner_config> configs;
    // for (int i = 0; i < devices.size(); i++)
    // {
    //     ax_parallel_runner_config config;
    //     config.dev_id = devices[i];

    //     char model_path[1024];
    //     sprintf(model_path, "tensor_test/llm_post/qwen2_r%d_post.axmodel", i);
    //     printf("model_path=%s\n", model_path);
    //     config.model_path = model_path;
    //     configs.push_back(config);
    // }

    ax_parallel_runner runner;
    if (runner.init("tensor_test/llm_post/qwen2_post.tar", devices) != 0)
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

    // axcl_Memcpy((void *)runner.get_input(0).phyAddr, buffer.data(), buffer.size(), AXCL_MEMCPY_HOST_TO_DEVICE, runner.get_devid());
    runner.set_input(buffer.data(), buffer.size(), 0);

    timer t_cost;
    for (size_t i = 0; i < DEFAULT_LOOP_COUNT; i++)
    {
        t_cost.start();
        runner.inference();
        printf("inference cost: %.2f ms\n", t_cost.cost());
        fflush(stdout);
    }

    for (size_t rankid = 0; rankid < devices.size(); rankid++)
    {
        int ret = axcl_Memcpy(runner.get_rank_input(rankid, 0).pVirAddr,
                              (void *)runner.get_rank_input(rankid, 0).phyAddr,
                              runner.get_rank_input(rankid, 0).nSize,
                              AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid(rankid));
        if (ret != 0)
        {
            printf("axcl_Memcpy failed, ret=%d\n", ret);
        }

        std::string dump_path = "dump_" + std::to_string(rankid) + ".bin";
        write_file(dump_path, runner.get_rank_input(rankid, 0).pVirAddr, runner.get_rank_input(rankid, 0).nSize);

        ret = memcmp(buffer.data(), runner.get_rank_input(rankid, 0).pVirAddr, buffer.size());
        if (ret != 0)
        {
            printf("input.bin is not equal to model input\n");
        }
    }

    std::vector<char> buffer_out = read_file("/home/axera/ax-llm/build/post_output.bin");
    if (buffer_out.size() != runner.get_rank_output(0, 0).nSize)
    {
        printf("output.bin size is not equal to model output size, model output size=%d, output.bin size=%ld\n", runner.get_rank_output(0, 0).nSize, buffer_out.size());
        return -1;
    }

    for (size_t rankid = 0; rankid < devices.size(); rankid++)
    {
        axcl_Memcpy(runner.get_rank_output(rankid, 0).pVirAddr,
                    (void *)runner.get_rank_output(rankid, 0).phyAddr,
                    runner.get_rank_output(rankid, 0).nSize,
                    AXCL_MEMCPY_DEVICE_TO_HOST, runner.get_devid(rankid));

        auto ret = memcmp(buffer_out.data(), runner.get_rank_output(rankid, 0).pVirAddr, buffer_out.size());
        if (ret != 0)
        {
            printf("output.bin is not equal to model output\n");
        }

        std::vector<float> buffer_out_fp32(buffer_out.size() / 2);
        std::vector<float> buffer_out_fp32_2(buffer_out.size() / 2);

        unsigned short *buffer_out_fp16 = (unsigned short *)buffer_out.data();
        unsigned short *buffer_out_fp16_2 = (unsigned short *)runner.get_rank_output(rankid, 0).pVirAddr;
        for (size_t i = 0; i < buffer_out.size() / 2; i++)
        {
            buffer_out_fp32[i] = bfloat16(buffer_out_fp16[i]).fp32();
            buffer_out_fp32_2[i] = bfloat16(buffer_out_fp16_2[i]).fp32();
        }
        printf("rankid=%d, cosine similarity=%f\n", rankid, consine_similarity(buffer_out_fp32.data(), buffer_out_fp32_2.data(), buffer_out_fp32.size()));
    }

    if (nullptr != p2p_handle)
    {
        axclrtDestoryP2PUnit(p2p_handle);
        printf("[INFO] p2p unit destroyed.\n");
    }

    runner.deinit();

    for (auto &devid : devices)
        axcl_Exit(devid);

    axclFinalize();

    return 0;
}