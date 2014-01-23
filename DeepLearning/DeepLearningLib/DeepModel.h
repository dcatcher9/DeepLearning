#pragma once
#include <amp.h>
#include <vector>
#include <random>

// for random number generator on GPU
#include "amp_tinymt_rng.h"

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
        // these vectors are initialzed before the corresponding array_views
        std::vector<float> value_;
        std::vector<float> expect_;
        std::vector<float> next_value_;
        std::vector<float> next_expect_;

    public:
        const int kMemorySize = 3;
        concurrency::array_view<float, 3> value_view_;
        concurrency::array_view<float, 3> expect_view_;
        concurrency::array_view<float, 3> next_value_view_;
        concurrency::array_view<float, 3> next_expect_view_;
        
        // seen data vectors, only lives in GPU
        std::vector<concurrency::array<float, 3>> memory_;

        tinymt_collection<3> rand_collection_;

    public:
        DataLayer(int depth, int width, int height, int seed = 0);
        // Disable copy constructor
        DataLayer(const DataLayer&) = delete;
        DataLayer(DataLayer&& other);

        void SetValue(const std::vector<float>& data);

        int depth() const
        {
            return value_view_.extent[0];
        }
        int width() const
        {
            return value_view_.extent[1];
        }
        int height() const
        {
            return value_view_.extent[2];
        }

        float ReconstructionError() const;
    };

    // Contains a collection of neurons, which is 3-dimensional according to data layer.
    // So the model layer has 4-dimensional structure. Responsible for processing data layer
    // using neurons within and adjusting neuron weights during learning.
    class ConvolveLayer
    {
    private:
        // this vector is initialized before weight_view_
        std::vector<float> weights_;
        // bias for visible nodes, i.e. bottom nodes
        std::vector<float> vbias_;
        std::vector<float> hbias_;

    public:
        concurrency::array_view<float, 4> weights_view_;
        concurrency::array<float, 4> weights_delta_;

        concurrency::array_view<float, 3> vbias_view_;
        concurrency::array<float, 3> vbias_delta_;

        concurrency::array_view<float> hbias_view_;
        concurrency::array<float> hbias_delta_;

    public:
        ConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

        int neuron_num() const
        {
            return weights_view_.extent[0];
        }
        int neuron_depth() const
        {
            return weights_view_.extent[1];
        }
        int neuron_width() const
        {
            return weights_view_.extent[2];
        }
        int neuron_height() const
        {
            return weights_view_.extent[3];
        }

        void PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
            DataLayer& top_layer, bool top_switcher) const;

        void PassDown(const DataLayer& top_layer, bool top_switcher,
            DataLayer& bottom_layer, bool bottom_switcher) const;

        void Train(const DataLayer& bottom_layer, const DataLayer& top_layer,
            float learning_rate, bool buffered_update);

        void ApplyBufferedUpdate(int buffer_size);

        void RandomizeParams(unsigned int seed);
    };

    // Pooling layer after convolvation, no params. 
    // Currently support max pooling, which is the most common pooling method.
    class PoolingLayer
    {
    public:
        int block_width_;
        int block_height_;

    public:
        PoolingLayer(int block_width, int block_height);

        void PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
            DataLayer& top_layer, bool top_switcher) const;

        void PassDown(const DataLayer& top_layer, bool top_switcher,
            DataLayer& bottom_layer, bool bottom_switcher) const;
    };

    class DeepModel
    {
    public:
        DeepModel(unsigned int model_seed = 0);
        // Disable copy constructor
        DeepModel(const DeepModel&) = delete;

        void AddDataLayer(int depth, int width, int height, int seed = 0);
        void AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height, 
            unsigned int rand_seed = 0);
        
        void PassUp(const std::vector<float>& data);
        void PassDown();

        float TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate);
        void TrainLayer(const std::vector<const std::vector<float>>& dataset,
            int layer_idx, int mini_batch_size, float learning_rate, int iter_count);

    private:
        std::vector<DataLayer> data_layers_;
        std::vector<ConvolveLayer> convolve_layers_;
        std::default_random_engine random_engine_;
    };
}

