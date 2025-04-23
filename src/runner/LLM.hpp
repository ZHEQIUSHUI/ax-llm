#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
#include "ax_sys_api.h"

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    // std::string template_prefill_filename_axmodel = "minicpmv/prefill_axmodel/minicpm_p96_l%d.axmodel";
    // int prefill_axmodel_num = 40;
    int prefill_token_num = 96;
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    std::string filename_image_encoder_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    int image_encoder_width = 448;
    int image_encoder_height = 448;

    TokenizerType tokenizer_type = TKT_LLaMa;
    std::string filename_tokenizer_model = "tokenizer.model";
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    bool b_use_mmap_load_embed = false;
    bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    bool b_use_topk = false;

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};

class LLM
{
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    LLMAttrType _attr;

    struct LLMLayer
    {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;

    ax_runner_ax650 image_encoder;

    int prefill_grpid = 1;
    int decode_grpid = 0;

    // std::vector<std::vector<unsigned short>> k_caches, v_caches;

    bool b_stop = false;

    static int FindMax(unsigned short *p, int n, float *val = 0)
    {
        float max_val = -MAXFLOAT;
        int max_index = 0;
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            float tmp = *reinterpret_cast<float *>(&proc);
            if (tmp > max_val)
            {
                max_val = tmp;
                max_index = i;
            }
        }

        // for (int i = 0; i < n; i += 4)
        // {
        //     uint16x4_t bf16_data = vld1_u16(&p[i]);
        //     uint32x4_t float_data = vmovl_u16(bf16_data);
        //     float32x4_t tmp_floats = vreinterpretq_f32_u32(vshlq_n_u32(float_data, 16));

        //     for (int j = 0; j < 4; j++)
        //     {
        //         float tmp = vgetq_lane_f32(tmp_floats, j);
        //         if (tmp > max_val)
        //         {
        //             max_val = tmp;
        //             max_index = i + j;
        //         }
        //     }
        // }

        if (val)
            *val = max_val;
        return max_index;
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 4, 32);
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

        llama_layers.resize(attr.axmodel_num);
        // prefill_layers.resize(attr.prefill_axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
                if (ret != 0)
                {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
            else
            {
                if (!attr.b_use_mmap_load_layer)
                {
                    if (!read_file(llama_layers[i].filename, llama_layers[i].layer_buffer_vec))
                    {
                        ALOGE("read_file(%s) failed", llama_layers[i].filename.c_str());
                        return false;
                    }
                }
                else
                {
                    llama_layers[i].layer_buffer.open_file(llama_layers[i].filename.c_str());
                }

                sprintf(axmodel_path, "read_file %s ok", llama_layers[i].filename.c_str());
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodedl.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init vpm axmodel(%s) failed", attr.filename_image_encoder_axmodedl.c_str());
            return false;
        }
        _attr.image_encoder_height = image_encoder.get_input(0).vShape[2];
        _attr.image_encoder_width = image_encoder.get_input(0).vShape[3];

        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 3, "count", axmodel_path);

        if (attr.b_dynamic_load_axmodel_layer)
        {
            // 加载第一层获取shape信息
            auto &layer = llama_layers[0];
            int ret;
            if (_attr.b_use_mmap_load_layer)
            {
                ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
            }
            else
            {
                ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
            }
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", layer.filename.c_str());
            }
        }

        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            printf("\n");
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            // auto &output_k_cache_out = llama_layers[0].layer.get_output("K_cache_out");
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(prefill_grpid, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            _attr.prefill_max_token_num = llama_layers[0].layer.get_input(llama_layers[0].layer.get_num_input_groups() - 1, "mask").vShape[2];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);

            ALOGI("image_encoder_height : %d, image_encoder_width: %d", _attr.image_encoder_height, _attr.image_encoder_width);
        }

        {
            bfloat16 bf16 = -65536.f;
            int prefill_split_num = _attr.prefill_max_token_num / _attr.prefill_token_num;
            std::vector<std::vector<unsigned short>> mask_p(prefill_split_num);
            for (size_t p = 0; p < prefill_split_num; p++)
            {
                std::vector<unsigned short> &mask_tmp = mask_p[p];
                mask_tmp.resize((p + 1) * _attr.prefill_token_num * _attr.prefill_token_num, bf16.data);

                size_t i = 0;
                for (size_t t = p * _attr.prefill_token_num; t < (p + 1) * _attr.prefill_token_num; t++)
                {
                    // if (t < input_embed_num)
                    {
                        for (size_t j = 0; j < p * _attr.prefill_token_num + i + 1; j++)
                            mask_tmp[i * ((p + 1) * _attr.prefill_token_num) + j] = 0;
                    }
                    i++;
                }
            }

            for (size_t p = 0; p < prefill_split_num; p++)
            {
                int prefill_grpid_tmp = p + 1;
                std::vector<unsigned short> &mask_tmp = mask_p[p];
                for (unsigned int m = 0; m < _attr.axmodel_num; m++)
                {
                    auto &layer = llama_layers[m];

                    auto &input_indices = layer.layer.get_input(prefill_grpid_tmp, "indices");
                    unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;

                    for (unsigned int i = 0; i < _attr.prefill_token_num; i++)
                    {
                        input_indices_ptr[i] = p * _attr.prefill_token_num + i;
                    }
                    // axcl_Memcpy((void *)input_indices.phyAddr, input_indices_ptr, _attr.prefill_token_num * sizeof(unsigned int), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());

                    auto &input_mask = layer.layer.get_input(prefill_grpid_tmp, "mask");
                    memcpy(input_mask.pVirAddr, mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));
                    // axcl_Memcpy((void *)input_mask.phyAddr, mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, layer.layer.get_devid());
                }
            }
        }

        if (attr.b_dynamic_load_axmodel_layer)
        {
            auto &layer = llama_layers[0];
            layer.layer.deinit();
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
        image_encoder.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> scale = {0.229, 0.224, 0.225};
        timer t;
        t.start();
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

        // std::vector<float> input_data(dst.rows * dst.cols * 3);

        float *input_data = (float *)image_encoder.get_input(0).pVirAddr;

        unsigned char *img_data = dst.data;
        int letterbox_rows = dst.rows;
        int letterbox_cols = dst.cols;

        for (int h = 0; h < letterbox_rows; h++)
        {
            for (int w = 0; w < letterbox_cols; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    input_data[out_index] = (float(img_data[in_index]) / 255.0 - mean[c]) / scale[c];
                }
            }
        }

        image_encoder.inference();
        int size = 1;
        for (size_t i = 0; i < image_encoder.get_output(0).vShape.size(); i++)
        {
            size *= image_encoder.get_output(0).vShape[i];
        }

        out_embed.resize(size);

        float *out_data = (float *)image_encoder.get_output(0).pVirAddr;

        for (size_t i = 0; i < size; i++)
        {
            out_embed[i] = bfloat16(out_data[i]).data;
        }

        // memcpy(out_embed.data(), image_encoder.get_output(0).pVirAddr, image_encoder.get_output(0).nSize);
        ALOGI("image encode time : %0.2f ms, size : %ld", t.cost(), out_embed.size());
        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        std::vector<int> input_ids = tokenizer->Encode(prompt, false);
        if (input_ids.size() > _attr.prefill_token_num)
        {
            ALOGE("input_ids(%d) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        // memcpy(out_embed.data() + 5 * _attr.tokens_embed_size, vpm_resampler.get_output("output").pVirAddr, vpm_resampler.get_output("output").nSize);

        return 0;
    }

    int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        std::vector<int> input_ids = tokenizer->Encode(prompt, true);

        // constexpr int IMG_CONTEXT = 151648; // InternVL2
        // constexpr int IMG_CONTEXT = 151667; // InternVL2.5
        constexpr int IMG_CONTEXT = 92546; // InternVL2.5-8B-MPO
        int offset = 0;
        int img_context_count = 0;

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            if (input_ids[i] == IMG_CONTEXT)
            {
                img_context_count++;
                if (img_context_count == 1)
                {
                    offset = i;
                }
            }
        }

        if (offset == 0)
        {
            ALOGE("offset == 0");
            return -1;
        }

        if (img_context_count != img_embed.size() / _attr.tokens_embed_size)
        {
            ALOGE("img_context_count(%d) != img_embed.size() / tokens_embed_size(%d)", img_context_count, img_embed.size() / _attr.tokens_embed_size);
            return -1;
        }

        // for (size_t i = 0; i < input_ids.size(); i++)
        // {
        //     printf("%d ", input_ids[i]);
        // }
        // printf("\n");

        if (input_ids.size() > _attr.prefill_max_token_num)
        {
            ALOGE("input_ids(%d) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++)
        {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        memcpy(out_embed.data() + offset * _attr.tokens_embed_size, img_embed.data(), img_embed.size() * sizeof(unsigned short));

        ALOGI("offset : %d, size : %d", offset, out_embed.size());
        return 0;
    }

    std::string Run(std::string input_str)
    {
        std::vector<unsigned short> test_embed;
        Encode(test_embed, input_str);
        return Run(test_embed);
    }

    std::string Run(std::vector<unsigned short> &test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        // std::vector<int> token_ids = tokenizer->Encode(input_str);
        // int len_of_input = token_ids.size();
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        ALOGI("input token num : %d", input_embed_num);
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("prefill_split_num : %d", prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            int prefill_grpid = p + 1;

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];
                auto &layer_llama = llama_layers[m];

                if (p > 0)
                {
                    auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                    auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");
                    for (size_t i = 0; i < p; i++)
                    {
                        auto &output_k_cache = layer.layer.get_output(i + 1, "K_cache_out");
                        memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + i * _attr.prefill_token_num * _attr.kv_cache_size,
                               output_k_cache.pVirAddr,
                               sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                        auto &output_v_cache = layer.layer.get_output(i + 1, "V_cache_out");
                        memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + i * _attr.prefill_token_num * _attr.kv_cache_size,
                               output_v_cache.pVirAddr,
                               sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);
                    }
                }

                auto &input_input = layer.layer.get_input(prefill_grpid, "input");
                memcpy(input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(prefill_grpid);

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
                auto &input_k_cache = layer_llama.layer.get_input(decode_grpid, "K_cache");
                memcpy((unsigned short *)input_k_cache.pVirAddr + p * _attr.prefill_token_num * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");
                AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
                auto &input_v_cache = layer_llama.layer.get_input(decode_grpid, "V_cache");
                memcpy((unsigned short *)input_v_cache.pVirAddr + p * _attr.prefill_token_num * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(prefill_grpid, "output");
                AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
                memcpy(embed_tmp.data(), output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        // ALOGI("prefill time cost: %.2f s", t_cost.cost() / 1000);

        // print token_ids
        // printf("%s\n", input_str.c_str());
        // for (size_t i = 0; i < token_ids.size(); i++)
        // {
        //     printf("%d ", token_ids[i]);
        // }
        // printf("\n");

        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);

        // memcpy(embed.data(),
        //        test_embed.data() + (input_embed_num - 1) * _attr.tokens_embed_size,
        //        _attr.tokens_embed_size * sizeof(unsigned short));

        {

            // post process
            auto &input = llama_post.get_input("input");
            memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();
            int max_index;
            if (_attr.b_use_topk)
            {
                AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                max_index = *(int *)llama_post.get_output("indices").pVirAddr;
            }
            else
            {
                auto &output_post = llama_post.get_output("output");
                AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
            }
            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = input_embed_num; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            // ALOGI("out %d %d", indices, next_token);
            embed_selector.getByIndex(next_token, embed);
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    int ret;
                    if (_attr.b_use_mmap_load_layer)
                    {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
                    }
                    else
                    {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
                    }
                    if (ret != 0)
                    {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    layer.layer.deinit();
                }
                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            {
                // post process
                auto &input = llama_post.get_input("input");
                memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                int max_index;
                if (_attr.b_use_topk)
                {
                    AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                    max_index = *(int *)llama_post.get_output("indices").pVirAddr;
                }
                else
                {
                    auto &output_post = llama_post.get_output("output");
                    AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                    unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                    float max_val = -MAXFLOAT;
                    max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                }
                next_token = max_index;

                if (tokenizer->isEnd(max_index))
                {
                    if (cached_token.size() && _attr.runing_callback)
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
        // token_ids.erase(token_ids.begin(), token_ids.begin() + len_of_input);

        final_out = tokenizer->Decode(token_ids);

        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            memset(llama_layers[i].layer.get_input(prefill_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "K_cache").nSize);
            memset(llama_layers[i].layer.get_input(prefill_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(prefill_grpid, "V_cache").nSize);
            memset(llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset(llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }
        return final_out;
    }
};
