#include "runner/utils/httplib.h"
#include "runner/utils/http_utils.hpp"
#include "runner/utils/json.hpp"
#include <iostream>
#include <fstream>

using json = nlohmann::json;
using namespace httplib;

const std::string SERVER_URL = "http://10.126.33.245:8000";

const std::string model = "qwen2-0.5B";

void call_generate()
{
    Client cli(SERVER_URL.c_str());
    json request = {
        {"model", model},
        {"prompt", "Hello!"}};

    auto res = cli.Post("/api/generate", request.dump(), "application/json");
    if (res && res->status == 200)
    {
        std::cout << "Generate response: " << res->body << std::endl;
    }
    else
    {
        std::cerr << "Generate request failed " << res->body << std::endl;
    }
}

void call_generate_streaming()
{
    Client cli(SERVER_URL.c_str());

    json request = {
        {"model", model},
        {"prompt", "Hello!"}};

    Headers headers = {
        {"Accept", "text/event-stream"},
        {"Content-Type", "application/json"}};

    auto res = cli.Post("/api/generate", headers, request.dump(), "application/json");

    if (res && res->status == 200)
    {
        std::istringstream stream(res->body);
        std::string line;
        while (std::getline(stream, line))
        {
            json chunk = json::parse(line, nullptr, false);
            std::string msg = chunk["response"];
            std::cout << msg;
            if (chunk["done"])
            {
                std::cout << std::endl;
            }
        }
    }
    else
    {
        std::cerr << "Streaming request failed " << std::endl;
    }
}

void call_chat()
{
    Client cli(SERVER_URL.c_str());
    json msg;
    msg["role"] = "user";
    msg["content"] = "Hello!";
    json request = {
        {"model", model},
        {"messages", {msg}}};

    auto res = cli.Post("/api/chat", request.dump(), "application/json");
    if (res && res->status == 200)
    {
        std::cout << "Chat response: " << res->body << std::endl;
    }
    else
    {
        std::cerr << "Chat request failed " << res->body << std::endl;
    }
}

void call_models()
{
    Client cli(SERVER_URL.c_str());
    auto res = cli.Get("/api/tags");
    if (res && res->status == 200)
    {
        std::cout << "Models: " << res->body << std::endl;
    }
    else
    {
        std::cerr << "Models request failed " << res->body << std::endl;
    }
}

void call_load()
{
    Client cli(SERVER_URL.c_str());
    json request = {
        {"model", model}};
    cli.set_read_timeout(5);
    auto res = cli.Post("/api/load", request.dump(), "application/json");
    if (res && res->status == 200)
    {
        std::cout << "Load response: " << res->body << std::endl;
    }
    else
    {
        std::cerr << "Load request failed " << res->body << std::endl;
    }
}

void call_upload(const std::string &file_path)
{
    Client cli(SERVER_URL.c_str());
    MultipartFormDataItems items = {
        {"image", file_path, "image/png", "test.png"} // 修改 MIME 类型以适应不同图片格式
    };

    auto res = cli.Post("/api/upload", items);
    if (res && res->status == 200)
    {
        std::cout << "Upload response: " << res->body << std::endl;
    }
    else
    {
        std::cerr << "Upload request failed " << res->body << std::endl;
    }
}

int main()
{
    call_models();
    call_load();
    call_chat();
    call_generate();
    call_generate_streaming();
    // call_upload("test.png"); // 替换为你的图片路径
    return 0;
}
