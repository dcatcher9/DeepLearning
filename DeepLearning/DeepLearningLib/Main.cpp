#include "DeepModel.h"
#include <iostream>

using namespace deep_learning_lib;

void main()
{
    DeepModel model;

    model.AddDataLayer(10, 1, 1, 1);
    model.AddConvolveLayer(5, 10, 1, 1);
    model.AddDataLayer(5, 1, 1, 2);

    std::vector<float> test = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
    
    for (int i = 0; i < 200; i++)
    {
        std::cout << "iter " << i << " : l2 err =" << model.TrainLayer(test, 0, 0.1f) << std::endl;
    }
}
