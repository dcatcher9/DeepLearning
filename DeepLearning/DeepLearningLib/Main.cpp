#include "DeepModel.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

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

    std::vector<const std::vector<float>> data;
    data.reserve(row_count);

    while (std::getline(ifs, line))
    {
        auto bits = from(split(line, " ", false)) >> take(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();
        data.emplace_back(bits);
    }

    DeepModel model;

    model.AddDataLayer(1, 16, 16, 1);
    model.AddConvolveLayer(20, 1, 8, 8);
    model.AddDataLayer(20, 9, 9, 2);

    /*for (int i = 0; i < 1000; i++)
    {
        float err = model.TrainLayer(data.front(), 0, 0.1f);
        std::cout << "iter = " << i << " err = " << err << std::endl;
    }*/
    
    model.TrainLayer(data, 0, 5, 0.2f, 0.5f, 1100);

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

    std::vector<const std::vector<float>> data;
    std::vector<const int> labels;
    data.reserve(row_count);
    labels.reserve(row_count);

    while (std::getline(ifs, line))
    {
        auto bits = split(line, " ", false);
        auto data_bits = from(bits) >> take(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();
        auto label_bits = from(bits) >> skip(row_len) >> select([](const std::string& s){return std::stof(s); }) >> to_vector();
        
        data.emplace_back(data_bits);
        for (int i = 0; i < label_bits.size(); i++)
        {
            if (label_bits[i] == 1.0f)
            {
                labels.emplace_back(i);
                break;
            }
        }
    }

    DeepModel model;

    model.AddDataLayer(256, 1, 1);
    model.AddConvolveLayer(100, 256, 1, 1);
    model.AddDataLayer(100, 1, 1, 2);
    model.AddOutputLayer(1, 10);

    /*for (int i = 0; i < 1000; i++)
    {
        float err = model.TrainLayer(data.front(), labels.front(), 0, 0.1f, 0.5f);
        std::cout << "iter = " << i << " err = " << err << std::endl;
    }*/

    model.TrainLayer(data, labels, 0, 5, 0.1f, 0.5f, 1000);

    model.GenerateImages("model_dump");
}

void main()
{
    //TestUSPS();
    TestRBM();
}
