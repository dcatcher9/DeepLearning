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
        const int kMemorySize = 3;
        concurrency::array_view<float, 3> data_view_;
        // data passed down by generative process, only lives in GPU
        concurrency::array<float, 3> data_generated_;
        // last data vectors, only lives in GPU
        std::vector<concurrency::array<float, 3>> memory_;

    public:
        DataLayer(int depth, int width, int height);
        // Disable copy constructor
        DataLayer(const DataLayer&) = delete;
        DataLayer(DataLayer&& other);

        void SetData(const std::vector<float>& data);

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
    class ConvolveLayer
    {
    private:
        // this vector is initialized before weight_view_
        std::vector<float> weights_;

    public:
        concurrency::array_view<float, 4> weight_view_;

    public:
        ConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

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

        void PassUp(concurrency::array_view<const float, 3> bottom_layer, 
            concurrency::array_view<float, 3> top_layer) const;

        void PassDown(concurrency::array_view<const float, 3> top_layer, 
            concurrency::array_view<float, 3> bottom_layer) const;

        void RandomizeParams(unsigned int seed);
    };

    class DeepModel
    {
    public:
        void AddDataLayer(int depth, int width, int height);
        void AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height, 
            unsigned int rand_seed = 0);
        
        void PassUp(const std::vector<float>& data);
        void PassDown();

    private:
        std::vector<DataLayer> data_layers_;
        std::vector<ConvolveLayer> convolve_layers_;
    };
}

