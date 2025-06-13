#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_model_runner/ax_parallel_runner.hpp"

// #include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "LLMPostprocess.hpp"

// #include <axcl.h>
// #include <axcl_rt_memory.h>
#include "utils/axcl_manager.h"
#include <axcl_rt_p2p.h>

#define ENABLE_COST 0
#if ENABLE_COST
#define COST(func)                                           \
    do                                                       \
    {                                                        \
        timer _timer;                                        \
        func;                                                \
        printf("%s cost: %3.2f ms\n", #func, _timer.cost()); \
    } while (0)
#else
#define COST(func) func
#endif

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    TokenizerType tokenizer_type = TKT_LLaMa;
    std::string filename_tokenizer_model = "tokenizer.model";
    bool b_bos = false, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    bool b_use_mmap_load_embed = false;
    // bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    std::string post_config_path = "post_config.json";

    std::vector<int> dev_ids = {0, 1, 2, 3};

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};

class LLM
{
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    AXCL_P2P_UNIT_HANDLE p2p_handle = nullptr;
    LLMAttrType _attr;

    struct LLMLayer
    {
        ax_parallel_runner layer;
#if HOST_DEBUG
        ax_runner_ax650_host layer_host;
#endif
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_parallel_runner llama_post;

    bool b_stop = false;

    LLMPostprocess postprocess;
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history, float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            logits[i] = *reinterpret_cast<float *>(&proc);
        }

        return postprocess.apply(logits, history);
    }

    std::vector<int> distributeModels(int cardCount, int modelCount)
    {
        std::vector<int> cardAssignments(modelCount);
        if (cardCount <= 0 || modelCount <= 0)
            return cardAssignments; // 返回空的或未初始化的 vector

        // 计算每张卡至少分配的模型数量
        int baseCount = modelCount / cardCount;
        // 计算余数，多出的模型会依次分配给前面的卡
        int remainder = modelCount % cardCount;

        int startIndex = 0;
        for (int card = 0; card < cardCount; ++card)
        {
            // 如果当前卡号在前 remainder 张卡中，则多分配一个模型
            int modelsOnThisCard = baseCount + (card < remainder ? 1 : 0);
            for (int i = 0; i < modelsOnThisCard; ++i)
            {
                cardAssignments[startIndex + i] = card;
            }
            startIndex += modelsOnThisCard;
        }

        return cardAssignments;
    }

public:
    bool Init(LLMAttrType attr)
    {

        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
        // test code
        // {
        //     std::vector<int> output;
        //     tokenizer.Encode("Today is National", output);
        //     // print output
        //     for (size_t i = 0; i < output.size(); i++)
        //     {
        //         printf("%d ", output[i]);
        //     }
        //     printf("\n");
        // }

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        printf("\n");
        // test code
        // {
        //     std::vector<unsigned short> embed = embed_selector.getByIndex(123);
        //     printf("embed size: %d\n", embed.size());
        //     for (int i = 0; i < embed.size(); i++)
        //     {
        //         bfloat16 bf16 = bfloat16(embed[i]);
        //         float val = bf16;
        //         printf("%d %0.22f\n", embed[i], val);
        //     }
        // }
        for (auto &devid : _attr.dev_ids)
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
            p2p_unit.u32DeviceNum = _attr.dev_ids.size();
            for (uint32_t i = 0; i < p2p_unit.u32DeviceNum; ++i)
            {
                p2p_unit.n32DeviceId[i] = device_list.devices[_attr.dev_ids[i]];
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

        llama_layers.resize(attr.axmodel_num);

        auto dev_assignments = distributeModels(_attr.dev_ids.size(), attr.axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), _attr.dev_ids);

            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = axcl_GetCMMRemain(_attr.dev_ids[dev_assignments[i]]);
            sprintf(axmodel_path, "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, _attr.dev_ids[dev_assignments[i]], remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);

            if (b_stop)
            {
                ALOGW("init axmodel(%s) stopped", llama_layers[i].filename.c_str());
                return false;
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), _attr.dev_ids);
        // int ret = llama_post.init("tensor_test/llm_post/qwen2_post.tar", _attr.dev_ids);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = axcl_GetCMMRemain(llama_post.get_devid());
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);
        printf("\n");
        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out_rank0").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache_rank0").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }
        }

        std::vector<int> v_remain_cmm;
        for (int i = 0; i < _attr.dev_ids.size(); i++)
        {
            v_remain_cmm.push_back(axcl_GetCMMRemain(_attr.dev_ids[i]));
        }
        printf(MACRO_PURPLE "________________________\n");
        printf("|%6s|%15s|\n", "ID", "remain cmm(MB)");
        printf("========================\n");
        for (int i = 0; i < _attr.dev_ids.size(); i++)
        {
            printf("|%6d|%15d|\n", _attr.dev_ids[i], v_remain_cmm[i]);
        }
        printf("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n" MACRO_END);

        if (!postprocess.load_config(attr.post_config_path))
        {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.release();
        }
        llama_post.release();

        embed_selector.Deinit();

        for (auto &devid : _attr.dev_ids)
            axcl_Exit(devid);

        if (nullptr != p2p_handle)
        {
            axclrtDestoryP2PUnit(p2p_handle);
            printf("[INFO] p2p unit destroyed.\n");
        }
    }

    void Stop()
    {
        b_stop = true;
    }

    std::string Run(std::string input_str)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        mask[_attr.kv_cache_num] = 0;
        std::vector<int> cached_token;
        std::vector<int> token_ids = tokenizer->Encode(input_str);
        int len_of_input = token_ids.size();
        timer t_cost;
        // print token_ids
        // printf("%s\n", input_str.c_str());
        // for (size_t i = 0; i < token_ids.size(); i++)
        // {
        //     printf("%d ", token_ids[i]);
        // }
        // printf("\n");

        int next_token = token_ids[0];
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        std::vector<unsigned short> embed;
        bool b_hit_eos = false;
        for (unsigned int indices = 0; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            embed_selector.getByIndex(next_token, embed);
            COST(llama_layers[0].layer.set_input("input", embed.data(), llama_layers[0].layer.get_input("input").nSize));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                std::function<void()> indices_mask_memcpy_func = [&]()
                {
                    layer.layer.set_input("indices", &indices, sizeof(indices)); // 0.28 ms

                    layer.layer.set_input("mask", mask.data(), mask.size() * sizeof(unsigned short)); // 0.28 ms
                };

                COST(indices_mask_memcpy_func());

                COST(layer.layer.inference()); // 1.45 ms
                std::function<void()> kvcache_memcpy_func = [&]()
                {
#pragma omp parallel for
                    for (int rankid = 0; rankid < _attr.dev_ids.size(); rankid++)
                    {
                        unsigned short *input_k_cache_ptr = (unsigned short *)layer.layer.get_rank_input(rankid, "K_cache_rank" + std::to_string(rankid)).phyAddr;
                        unsigned short *input_v_cache_ptr = (unsigned short *)layer.layer.get_rank_input(rankid, "V_cache_rank" + std::to_string(rankid)).phyAddr;

                        axcl_Memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size,
                                    (void *)layer.layer.get_rank_output(rankid, "K_cache_out_rank" + std::to_string(rankid)).phyAddr,
                                    sizeof(unsigned short) * _attr.kv_cache_size,
                                    AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid(rankid));

                        axcl_Memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size,
                                    (void *)layer.layer.get_rank_output(rankid, "V_cache_out_rank" + std::to_string(rankid)).phyAddr,
                                    sizeof(unsigned short) * _attr.kv_cache_size,
                                    AXCL_MEMCPY_DEVICE_TO_DEVICE, layer.layer.get_devid(rankid));
                    }
                };

                COST(kvcache_memcpy_func()); // 0.44 ms
                std::function<void()> io_memcpy_func = [&]()
                {
                    if (m == _attr.axmodel_num - 1)
                    {
                        axcl_Memcpy((void *)layer.layer.get_output("output").pVirAddr,
                                    (void *)layer.layer.get_output("output").phyAddr, layer.layer.get_output("output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                        llama_post.set_input(0, layer.layer.get_output("output").pVirAddr, layer.layer.get_output("output").nSize);
                    }
                    else if (m < _attr.axmodel_num - 1)
                    {
                        axcl_Memcpy((void *)layer.layer.get_output("output").pVirAddr,
                                    (void *)layer.layer.get_output("output").phyAddr, layer.layer.get_output("output").nSize, AXCL_MEMCPY_DEVICE_TO_HOST, layer.layer.get_devid());

                        llama_layers[m + 1].layer.set_input("input", layer.layer.get_output("output").pVirAddr, layer.layer.get_output("output").nSize);
                    }
                };

                COST(io_memcpy_func()); // 0.46 ms
            }

            mask[indices] = 0;
            if (indices + 1 < token_ids.size())
            {
                next_token = token_ids[indices + 1];
            }
            else
            {
                // post process
                COST(llama_post.inference()); // 6.6 ms
                auto &output_post = llama_post.get_output("output");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                COST(axcl_Memcpy(post_out, (void *)output_post.phyAddr, output_post.nSize, AXCL_MEMCPY_DEVICE_TO_HOST, llama_post.get_devid())); // 1 ms

                int max_index;
                std::function<void()> post_process_func = [&]()
                {
                    max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
                    next_token = max_index;
                };
                COST(post_process_func()); // 0.4 ms

                if (tokenizer->isEnd(max_index))
                {
                    if (cached_token.size())
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback)
                {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3)
                    {
                        float t_cost_ms = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec, _attr.reserve);
                        cached_token.clear();
                    }
                }
            }
            if (_attr.runing_callback == nullptr)
                update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos)
            {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        // 去掉 len_of_input 那部分
        token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};
