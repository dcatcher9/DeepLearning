#include "DeepModel.h"

using namespace deep_learning_lib;

void main()
{
    DeepModel model;

    model.AddDataLayer(10, 1, 1, 1);
    model.AddConvolveLayer(5, 10, 1, 1);
    model.AddDataLayer(5, 1, 1, 2);

    std::vector<float> test = { 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 };
    model.PassUp(test);
    model.PassDown();
}
