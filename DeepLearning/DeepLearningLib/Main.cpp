#include "DeepModel.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "cpplinq.hpp"

using namespace deep_learning_lib;

std::vector<std::string> split(const std::string& s, const std::string& delim, const bool keep_empty = true) {
    std::vector<std::string> result;
    if (delim.empty()) {
        result.push_back(s);
        return result;
    }
    std::string::const_iterator substart = s.begin(), subend;
    while (true) {
        subend = search(substart, s.end(), delim.begin(), delim.end());
        std::string temp(substart, subend);
        if (keep_empty || !temp.empty()) {
            result.push_back(temp);
        }
        if (subend == s.end()) {
            break;
        }
        substart = subend + delim.size();
    }
    return result;
}

void TestUSPS()
{
    using namespace cpplinq;

    std::ifstream ifs(".\\Data Files\\usps_all.txt");

    std::string line;
    std::getline(ifs, line);

    std::vector<std::string> headers = split(line, " ");
    int row_count = std::stoi(headers[0]);
    int row_len = std::stoi(headers[1]);

    const float train_fraction = 0.8f;

    std::vector<const std::vector<float>> train_data;
    std::vector<const int> train_labels;

    std::vector<const std::vector<float>> test_data;
    std::vector<const int> test_labels;

    train_data.reserve(row_count);
    train_labels.reserve(row_count);
    test_data.reserve(row_count);
    test_labels.reserve(row_count);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> rand;

    while (std::getline(ifs, line))
    {
        auto bits = split(line, " ", false);
        auto data_bits = from(bits) >> take(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();
        auto label_bits = from(bits) >> skip(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();

        if (rand(generator) < train_fraction)
        {
            train_data.emplace_back(data_bits);
            for (int i = 0; i < label_bits.size(); i++)
            {
                if (label_bits[i] == 1.0f)
                {
                    train_labels.emplace_back(i);
                    break;
                }
            }
        }
        else
        {
            test_data.emplace_back(data_bits);
            for (int i = 0; i < label_bits.size(); i++)
            {
                if (label_bits[i] == 1.0f)
                {
                    test_labels.emplace_back(i);
                    break;
                }
            }
        }
    }

    DeepModel model;

    model.AddDataLayer(1, 16, 16);
    model.AddConvolveLayer(200, 8, 8);
    model.AddDataLayer();
    model.AddOutputLayer(10);

    const float dropout_prob = 0.5f;
    std::uniform_int_distribution<size_t> index_rand(0, train_data.size() - 1);

    for (int i = 0; i < 500; i++)
    {
        size_t idx = index_rand(generator);
        const auto& data = train_data[idx];
        const auto label = train_labels[idx];
        model.TrainLayer(data, 1, 0.2f, dropout_prob, label, false);
        float precision = model.Evaluate(test_data, test_labels, 0, dropout_prob);
        std::cout << "P = " << precision << std::endl;
    }


    model.GenerateImages("model_dump");
}

void TestRBM()
{
    using namespace cpplinq;

    std::ifstream ifs(".\\Data Files\\usps_all.txt");

    std::string line;
    std::getline(ifs, line);

    std::vector<std::string> headers = split(line, " ");
    int row_count = std::stoi(headers[0]);
    int row_len = std::stoi(headers[1]);

    const float train_fraction = 0.9f;

    std::vector<const std::vector<float>> train_data;
    std::vector<const int> train_labels;

    std::vector<const std::vector<float>> test_data;
    std::vector<const int> test_labels;

    train_data.reserve(row_count);
    train_labels.reserve(row_count);
    test_data.reserve(row_count);
    test_labels.reserve(row_count);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> rand;

    while (std::getline(ifs, line))
    {
        auto bits = split(line, " ", false);
        auto data_bits = from(bits) >> take(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();
        auto label_bits = from(bits) >> skip(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();

        if (rand(generator) < train_fraction)
        {
            train_data.emplace_back(data_bits);
            for (int i = 0; i < label_bits.size(); i++)
            {
                if (label_bits[i] == 1.0f)
                {
                    train_labels.emplace_back(i);
                    break;
                }
            }
        }
        else
        {
            test_data.emplace_back(data_bits);
            for (int i = 0; i < label_bits.size(); i++)
            {
                if (label_bits[i] == 1.0f)
                {
                    test_labels.emplace_back(i);
                    break;
                }
            }
        }
    }

    DeepModel model;

    model.AddDataLayer(256, 1, 1);
    model.AddConvolveLayer(200, 1, 1);
    model.AddDataLayer();
    model.AddOutputLayer(10);

    const float dropout_prob = 0.5f;
    std::uniform_int_distribution<size_t> index_rand(0, train_data.size() - 1);

    for (int i = 0; i < 500; i++)
    {
        size_t idx = index_rand(generator);
        const auto& data = train_data[idx];
        const auto label = train_labels[idx];
        model.TrainLayer(data, 1, 0.2f, dropout_prob, label, false);
        float precision = model.Evaluate(test_data, test_labels, 0, dropout_prob);
        std::cout << "P = " << precision << std::endl;
    }

    model.GenerateImages("model_dump");
}

void main()
{
    TestUSPS();
    //TestRBM();
}
