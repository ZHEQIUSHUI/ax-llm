#include "runner/ax_model_runner/ax_parallel_runner.hpp"
#include <axcl_rt_memory.h>
#include <axcl.h>
#include "cmdline.hpp"
#include "memory_utils.hpp"
#include "bfloat16.hpp"
#include <axcl_rt_p2p.h>

#include <cmath>
#include <timer.hpp>
#include <unordered_map>

static std::unordered_map<std::string, std::vector<double>> cost_records;
void print_cost_statistics()
{
    for (const auto &pair : cost_records)
    {
        const std::string &func_name = pair.first;
        std::vector<double> costs = pair.second;

        double sum = 0.0;
        double max_cost = costs[0];
        double min_cost = costs[0];
        double mid_cost = 0.0;
        for (double cost : costs)
        {
            sum += cost;
        }
        max_cost = *std::max_element(costs.begin(), costs.end());
        min_cost = *std::min_element(costs.begin(), costs.end());
        std::sort(costs.begin(), costs.end());
        mid_cost = costs[costs.size() / 2];

        double avg_cost = sum / costs.size();

        printf("total:%3d, avg: %5.2f ms, max: %5.2f ms, min: %5.2f ms, mid: %5.2f ms %s\n",
               costs.size(), avg_cost, max_cost, min_cost, mid_cost, func_name.c_str());
    }
}

#define COST(func)                           \
    do                                       \
    {                                        \
        timer _timer;                        \
        func;                                \
        double cost = _timer.cost();         \
        cost_records[#func].push_back(cost); \
    } while (0);

static void print_io_info(std::vector<ax_runner_tensor_t> &input, std::vector<ax_runner_tensor_t> &output)
{
    printf("  input size: %ld\n", input.size());
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = input[i];
        printf("      name: \e[1;32m%8s", info.sName.c_str());

        std::string dt = "unknown";

        printf(" \e[1;31m[%s] ", dt.c_str());

        std::string ct = "unknown";

        printf("\e[1;31m[%s]", ct.c_str());

        printf(" \n          \e[1;31m");

        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n");
    }

    printf("  output size: %ld\n", output.size());
    for (uint32_t i = 0; i < output.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = output[i];
        printf("      name: \e[1;32m%8s \e[0m\n          \e[1;31m", info.sName.c_str());
        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n");
    }
}

int main(int argc, char **argv)
{
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "model path", true);
    parser.add<int>("parallel", 'p', "parallel num", false, 4);
    parser.add<int>("loop", 'l', "loop num", false, 1);
    parser.parse_check(argc, argv);
    std::string model_path = parser.get<std::string>("model");
    int parallel_num = parser.get<int>("parallel");
    int loop_num = parser.get<int>("loop");

    std::vector<int> devices;

    printf("parallel num: %d devices: [", parallel_num);
    for (int i = 0; i < parallel_num; ++i)
    {
        devices.push_back(i);
        printf("%d", i);
        if (i < parallel_num - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");

    AXCL_P2P_UNIT_HANDLE p2p_handle = nullptr;
    auto ret = axclInit(nullptr);
    if (0 != ret)
    {
        return ret;
    }

    for (auto &devid : devices)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return false;
        }
    }

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

    ax_parallel_runner runner;
    runner.init(model_path, devices);

    for (int grpid = 0; grpid < runner.get_num_groups(); grpid++)
    {
        std::vector<ax_runner_tensor_t> input, output;
        for (int i = 0; i < runner.get_num_inputs(); i++)
        {
            input.push_back(runner.get_input(grpid, i));
        }
        for (int i = 0; i < runner.get_num_outputs(); i++)
        {
            output.push_back(runner.get_output(grpid, i));
        }
        printf("\n\ngroup %d:\n", grpid);
        print_io_info(input, output);
    }

    // warmup
    for (int i = 0; i < 5; i++)
    {
        runner.inference();
    }

    for (int i = 0; i < loop_num; i++)
    {
        COST(runner.inference());
    }

    print_cost_statistics();

    runner.deinit();

    axclrtDestoryP2PUnit(p2p_handle);

    for (auto &devid : devices)
    {
        axcl_Exit(devid);
    }

    axclFinalize();

    return 0;
}