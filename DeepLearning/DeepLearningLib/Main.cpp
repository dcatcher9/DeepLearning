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
        break;
    }

    DeepModel model;

    model.AddDataLayer(1, 16, 16, 1);
    model.AddConvolveLayer(20, 1, 8, 8);
    model.AddDataLayer(20, 8, 8, 2);

    for (int i = 0; i < 1000; i++)
    {
        float err = model.TrainLayer(data.front(), 0, 0.1f);
        std::cout << "iter = " << i << " err = " << err << std::endl;
    }
    
    //model.TrainLayer(data, 0, 1, 0.1f, 10000);

}

void TestRBM()
{
    DeepModel model;

    model.AddDataLayer(10, 1, 1);
    model.AddConvolveLayer(20, 10, 1, 1);
    model.AddDataLayer(20, 1, 1, 2);

    std::vector<float> data = { 1, 1, 0, 0, 0, 1, 1, 0, 0, 0 };
    for (int i = 0; i < 1000; i++)
    {
        float err = model.TrainLayer(data, 0, 0.1f);
        std::cout << "iter = " << i << " err = " << err << std::endl;
    }
}

void main()
{
    //TestUSPS();
    TestRBM();
}
