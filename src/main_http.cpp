#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <queue>
#include <signal.h>

#include "runner/utils/httplib.h"
#include "runner/utils/json.hpp"
#include "runner/utils/string_utility.hpp"
#include "runner/LLM.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <glob.h>
#endif
#include <future>

std::vector<std::string> glob(const std::string &pattern)
{
    std::vector<std::string> results;

#ifdef _WIN32
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &findFileData);

    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                results.push_back(findFileData.cFileName);
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
#else
    glob_t glob_result;
    if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0)
    {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        {
            results.emplace_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);
    }
#endif

    return results;
}

httplib::Server svr;
const int PORT = 8000;
const std::string UPLOAD_DIR = "uploads/";
const std::string MODEL_DIR = "models/";

static std::queue<std::string> g_msg_queue;

void __sigExit(int iSigNo)
{
    svr.stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
    g_msg_queue.push(p_str);
}

template <typename T>
T getAttr(nlohmann::json &json, const std::string &key, T default_value)
{
    if (json.contains(key))
    {
        return json[key].get<T>();
    }
    return default_value;
}

class Worker
{
public:
    LLM gllm;
    std::atomic<bool> gllm_runing = false;
    bool gllm_init = false;
    bool gllm_initing = false;

private:
    using Task = std::function<void()>;
    std::thread worker_thread;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;

    void run()
    {
        while (true)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this]
                               { return stop_flag || !tasks.empty(); });
                if (stop_flag && tasks.empty())
                {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            task(); // 执行任务
        }
    }

    // **支持无参数任务**
    void addTask(Task task)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

    // 模板接口：添加任务并返回 std::future 用于获取返回值
    template <typename F, typename... Args>
    auto addTaskWithResult(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using result_type = typename std::result_of<F(Args...)>::type;
        // 将函数及其参数绑定成一个无参函数
        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<result_type> res = task->get_future();
        // 将任务封装为 lambda，确保在工作线程中执行
        addTask([task]()
                { (*task)(); });
        return res;
    }

public:
    Worker() : stop_flag(false) {}

    ~Worker()
    {
        Stop();
    }

    bool Run()
    {
        worker_thread = std::thread(&Worker::run, this);
        return true;
    }

    void Stop()
    {
        if (!stop_flag)
        {
            stop_flag = true;
            condition.notify_one();
            if (worker_thread.joinable())
            {
                worker_thread.join();
            }
        }
    }

    // **支持带参数的任务**
    void RunAsync(std::string prompt)
    {

        addTask([this, prompt]()
                { RunSync(prompt, llm_running_callback); });
    }

    bool Load(std::string model)
    {
        // addTask([this, model]()
        //         { LoadModel(model); });
        // return true;
        auto future_result = addTaskWithResult(&Worker::LoadModel, this, model);
        return future_result.get();
    }

    std::string RunSync(std::string prompt, LLMRuningCallback cb)
    {
        if (gllm_runing)
        {
            return "";
        }
        gllm_runing = true;
        gllm.getAttr()->runing_callback = cb;
        auto output = gllm.Run(prompt);
        gllm_runing = false;
        // std::cout << "Chat result: " << output << std::endl;
        return output;
    }

    bool LoadModel(const std::string &model_name)
    {
        if (gllm_init)
        {
            return true;
        }
        if (gllm_initing)
        {
            ALOGE("loading,plz wait");
            return false;
        }
        std::string model_path = MODEL_DIR + model_name + ".json";
        if (!file_exist(model_path))
        {
            ALOGE("model file not exist: %s", model_path.c_str());
            return false;
        }
        nlohmann::json model_json;
        try
        {
            model_json = nlohmann::json::parse(std::ifstream(model_path));
        }
        catch (const std::exception &e)
        {
            ALOGE("parse model json error: %s", e.what());
            return false;
        }

        LLMAttrType attr;
        attr.template_filename_axmodel = getAttr<std::string>(model_json, "template_filename_axmodel", attr.template_filename_axmodel);
        attr.filename_post_axmodel = getAttr<std::string>(model_json, "filename_post_axmodel", attr.filename_post_axmodel);
        attr.tokenizer_type = (TokenizerType)getAttr<int>(model_json, "tokenizer_type", attr.tokenizer_type);
        attr.filename_tokenizer_model = getAttr<std::string>(model_json, "filename_tokenizer_model", attr.filename_tokenizer_model);
        attr.filename_tokens_embed = getAttr<std::string>(model_json, "filename_tokens_embed", attr.filename_tokens_embed);
        attr.b_bos = getAttr<bool>(model_json, "bos", attr.b_bos);
        attr.b_eos = getAttr<bool>(model_json, "eos", attr.b_eos);
        attr.axmodel_num = getAttr<int>(model_json, "axmodel_num", attr.axmodel_num);
        attr.tokens_embed_num = getAttr<int>(model_json, "tokens_embed_num", attr.tokens_embed_num);
        attr.tokens_embed_size = getAttr<int>(model_json, "tokens_embed_size", attr.tokens_embed_size);
        attr.b_use_mmap_load_embed = getAttr<bool>(model_json, "use_mmap_load_embed", attr.b_use_mmap_load_embed);
        attr.post_config_path = getAttr<std::string>(model_json, "post_config_path", attr.post_config_path);
        gllm_initing = true;
        if (!gllm.Init(attr))
        {
            return false;
        }
        gllm_initing = false;
        gllm_init = true;
        return true;
    }
};

Worker worker;

bool check_model_available(const httplib::Request &req, httplib::Response &res)
{
    if (!worker.gllm_init)
    {
        if (worker.gllm_initing)
        {
            ALOGE("model initing");
            res.status = 400;
            res.set_content("{\"error\": \"Model initing\"}", "application/json");
            return false;
        }
        else
        {
            ALOGE("model not init");
            res.status = 400;
            res.set_content("{\"error\": \"Model not init\"}", "application/json");
            return false;
        }
    }

    if (worker.gllm_runing)
    {
        res.status = 400;
        res.set_content("{\"error\": \"llm is running\"}", "application/json");
        return false;
    }

    return true;
}

void set_llm_config(nlohmann::json &body)
{
    if (body.contains("temperature"))
    {
        float temperature = body["temperature"];
        ALOGI("temperature: %f", temperature);
        if (temperature > 0)
        {
            worker.gllm.getPostprocess()->set_temperature(true, temperature);
        }
        else
        {
            ALOGE("temperature: %f is invalid", temperature);
            worker.gllm.getPostprocess()->set_temperature(false, temperature);
        }
    }
    if (body.contains("repetition_penalty"))
    {
        float repetition_penalty = body["repetition_penalty"];
        ALOGI("repetition_penalty: %f", repetition_penalty);
        if (repetition_penalty - 1 < 0.0001)
        {
            ALOGI("repetition_penalty: %f is skip", repetition_penalty);
        }
        else
        {
            worker.gllm.getPostprocess()->set_repetition_penalty(true, repetition_penalty);
        }
    }
    if (body.contains("top-p"))
    {
        float top_p = body["top-p"];
        ALOGI("top-p: %f", top_p);
        if (top_p > 0 && top_p < 1)
        {
            worker.gllm.getPostprocess()->set_top_p_sampling(true, top_p);
        }
        else
        {
            ALOGE("top-p: %f is invalid", top_p);
            worker.gllm.getPostprocess()->set_top_p_sampling(false, top_p);
        }
    }
    if (body.contains("top-k"))
    {
        int top_k = body["top-k"];
        ALOGI("top-k: %d", top_k);
        if (top_k > 0)
        {
            worker.gllm.getPostprocess()->set_top_k_sampling(true, top_k);
        }
        else
        {
            ALOGE("top-k: %d is invalid", top_k);
            worker.gllm.getPostprocess()->set_top_k_sampling(false, top_k);
        }
    }
}

bool content_provider(size_t offset, httplib::DataSink &sink)
{
    // for (int i = 0; i < 5; ++i) {  // 模拟逐步生成文本
    //     nlohmann::json chunk;
    //     chunk["response"] = "Chunk " + std::to_string(i + 1);
    //     chunk["done"] = (i == 4);

    //     std::string data = "data: " + chunk.dump() + "\n\n";
    //     sink.write(data.data(), data.size());

    //     std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 模拟延迟
    // }
    // sink.done();

    while (worker.gllm_runing)
    {
        if (g_msg_queue.size())
        {
            auto str = g_msg_queue.back();
            g_msg_queue.pop();

            nlohmann::json chunk;
            chunk["response"] = str;
            chunk["done"] = false;

            auto msg = chunk.dump();
            msg += "\n";
            sink.write(msg.data(), msg.size());
        }
        usleep(1000);
    }
    std::stringstream oss;
    while (g_msg_queue.size())
    {
        auto str = g_msg_queue.back();
        g_msg_queue.pop();
        oss << str;
    }
    auto tmp = oss.str();
    // sink.write(tmp.data(), tmp.size());

    nlohmann::json chunk;
    chunk["response"] = tmp;
    chunk["done"] = true;

    auto msg = chunk.dump();
    msg += "\n";
    sink.write(msg.data(), msg.size());

    sink.done();
    return true;
}

void handle_generate(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.contains("model") || !body.contains("prompt"))
    {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    set_llm_config(body);

    std::string prompt = body["prompt"];
    ALOGI("%s", prompt.c_str());
    worker.RunAsync(prompt);

    res.set_chunked_content_provider("text/event-stream", content_provider);
}

void handle_chat(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);

    if (body.is_discarded() || !body.contains("model") || !body.contains("messages"))
    {
        ALOGE("Invalid request format");
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }
    if (!check_model_available(req, res))
    {
        ALOGE("model not available");
        return;
    }

    set_llm_config(body);

    std::vector<nlohmann::json> messages = body["messages"];

    for (auto &message : messages)
    {
        if (message.contains("role") && message.contains("content"))
        {
            auto output = worker.RunSync(message["content"].get<std::string>(), nullptr);

            nlohmann::json response;
            response["message"] = output;
            response["model"] = body["model"];
            response["done"] = true;

            res.status = 200;
            res.set_content(response.dump(), "application/json");

            return;
        }
    }

    res.status = 400;
    res.set_content("{\"error\": \"Invalid message format\"}", "application/json");
    return;
}

std::vector<nlohmann::json> get_models()
{
    std::vector<nlohmann::json> models;
    std::vector<std::string> model_files = glob(MODEL_DIR + "*.json");
    for (const auto &file : model_files)
    {
        nlohmann::json model;
        std::vector<std::string> parts = string_utility<std::string>::split(file, "/");
        std::string filename = parts[parts.size() - 1];
        std::string name = filename.substr(0, filename.length() - 5);
        model["name"] = name;
        model["path"] = file;
        models.push_back(model);
    }
    return models;
}

void handle_models(const httplib::Request &req, httplib::Response &res)
{
    nlohmann::json response;

    // std::vector<std::string> model_files = glob(MODEL_DIR + "*.json");
    // for (const auto &file : model_files)
    // {
    //     nlohmann::json model;
    //     std::vector<std::string> parts = string_utility<std::string>::split(file, "/");
    //     std::string filename = parts[parts.size() - 1];
    //     std::string name = filename.substr(0, filename.length() - 5);
    //     model["name"] = name;
    //     response["models"].push_back(model);
    // }

    auto models = get_models();

    for (auto &model : models)
    {
        response["models"].push_back(model);
    }

    res.set_content(response.dump(), "application/json");
}

void handle_load(const httplib::Request &req, httplib::Response &res)
{
    auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.contains("model"))
    {
        ALOGE("Invalid request format");
        res.status = 400;
        res.set_content("{\"error\": \"Invalid request format\"}", "application/json");
        return;
    }

    worker.Load(body["model"]);
    //     ALOGE("check_model failed");
    //     res.status = 400;
    //     res.set_content("{\"error\": \"Invalid model\"}", "application/json");
    //     return;
    // }

    res.set_content("{\"message\": \"load model success\"}", "application/json");
}

void handle_upload(const httplib::Request &req, httplib::Response &res)
{
    const auto &file = req.get_file_value("image");
    std::string file_path = UPLOAD_DIR + file.filename;

    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs)
    {
        res.status = 500;
        res.set_content("{\"error\": \"Failed to save file\"}", "application/json");
        return;
    }
    ofs.write(file.content.data(), file.content.size());
    ofs.close();

    nlohmann::json response;
    response["message"] = "File uploaded successfully";
    response["file_path"] = file_path;
    res.set_content(response.dump(), "application/json");
}

int main()
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    auto models = get_models();
    printf(MACRO_GREEN "\nmodels list:\n");
    for (auto &model : models)
    {
        printf("     %s\n", model.dump().c_str());
    }
    printf(MACRO_END "\n");

    worker.Run();

    svr.Get("/api/tags", handle_models);
    svr.Post("/api/load", handle_load);
    svr.Post("/api/generate", handle_generate);
    svr.Post("/api/chat", handle_chat);
    svr.Post("/api/upload", handle_upload);

    svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) -> httplib::Server::HandlerResponse
                                {
                                    res.set_header("Access-Control-Allow-Origin", "*");
                                    if (req.method == "OPTIONS")
                                    {
                                        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                                        res.set_header("Access-Control-Allow-Headers", "Content-Type");
                                        res.status = 200;
                                        return httplib::Server::HandlerResponse::Handled; // 表示已处理，不再继续
                                    }
                                    return httplib::Server::HandlerResponse::Unhandled; // 继续处理请求
                                });

    std::cout << "Server running on port " << PORT << "..." << std::endl;
    svr.listen("0.0.0.0", PORT);
    worker.Stop();
    worker.gllm.Deinit();
    return 0;
}
