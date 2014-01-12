#include "DeepModel.h"

using namespace deep_learning_lib;

void main()
{
    DeepModel model;

    model.AddDataLayer(10, 1, 1);
    model.AddConvolveLayer(5, 1, 1, 1);
    model.AddDataLayer(5, 1, 1);
}