#pragma once
#include <amp.h>
#include <vector>

namespace deep_learning_lib
{
    // 3-dimensional data layer, cache the intermediate result in neural network
    // 3 dimension:
    //     _____________________
    //    /                    /|
    //   / height             / |
    //  /       width        /  |
    //  ---------------------   |
    //  |                   |   |
    //  | depth             |  /
    //  |                   | / 
    //  ---------------------/
    //  depth dimension is orderless, representing the concept of unordered set.
    class DataLayer
    {
    private:
        // this vector is initialzed before data_view_
        std::vector<float> data_;

    public:
        concurrency::array_view<float, 3> data_view_;

    public:
        DataLayer(int depth, int width, int height);

        int depth() const
        {
            return data_view_.extent[0];
        }
        int width() const
        {
            return data_view_.extent[1];
        }
        int height() const
        {
            return data_view_.extent[2];
        }
    };

    // Contains a collection of neurons, which is 3-dimensional according to data layer.
    // So the model layer has 4-dimensional structure. Responsible for processing data layer
    // using neurons within and adjusting neuron weights during learning.
    class ModelLayer
    {
    private:
        // this vector is initialized before weight_view_
        std::vector<float> weights_;

    public:
        concurrency::array_view<float, 4> weight_view_;

    public:
        ModelLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height);

        int neuron_num() const
        {
            return weight_view_.extent[0];
        }
        int neuron_depth() const
        {
            return weight_view_.extent[1];
        }
        int neuron_width() const
        {
            return weight_view_.extent[2];
        }
        int neuron_height() const
        {
            return weight_view_.extent[3];
        }

        void PassUp(const DataLayer& bottom_layer, DataLayer& top_layer) const;
        void PassDown(const DataLayer& top_layer, DataLayer& bottom_layer) const;
    };

    class DeepModel
    {
    public:

    };
}

