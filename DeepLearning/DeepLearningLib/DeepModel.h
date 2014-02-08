#pragma once
#include <amp.h>
#include <vector>
#include <random>

// for random number generator on GPU
#include "amp_tinymt_rng.h"

// for bitmap generation
#include "bitmap_image.hpp"

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
        std::vector<float>  value_;
        std::vector<float>  expect_;

        std::vector<float>  next_value_;
        std::vector<float>  next_expect_;

        float active_prob_;
        std::vector<int>    active_;
        
    public:
        const int kMemorySize = 3;
        concurrency::array_view<float, 3>   value_view_;
        concurrency::array_view<float, 3>   expect_view_;
        concurrency::array_view<float, 3>   next_value_view_;
        concurrency::array_view<float, 3>   next_expect_view_;
        // for dropout
        concurrency::array_view<int, 3>     active_view_;
        
        // seen data vectors, only lives in GPU
        std::vector<concurrency::array<float, 3>> memory_;

        tinymt_collection<3> rand_collection_;

    public:
        DataLayer(int depth, int height, int width, int seed = 0);
        // Disable copy constructor
        DataLayer(const DataLayer&) = delete;
        DataLayer(DataLayer&& other);

        void SetValue(const std::vector<float>& data);

        inline int depth() const
        {
            return value_view_.extent[0];
        }
        inline int height() const
        {
            return value_view_.extent[1];
        }
        inline int width() const
        {
            return value_view_.extent[2];
        }

        void Activate(float probability = 1.0f);

        float ReconstructionError() const;

        bitmap_image GenerateImage() const;
    };

    // Currently support 1-of-N classifier output.
    // It contains both data and weight parameters. 
    // Support both discriminative and generative training.
    class OutputLayer
    {
    private:
        std::vector<float> outputs_;
        std::vector<float> next_outputs_;
        std::vector<float> bias_;
        std::vector<float> weights_;

    public:
        concurrency::array_view<float>  outputs_view_;
        concurrency::array_view<float>  next_outputs_view_;
        
        concurrency::array_view<float>  bias_view_;
        concurrency::array<float>       bias_delta_;

        concurrency::array_view<float, 4>   weights_view_;
        concurrency::array<float, 4>        weights_delta_;

    public:
        OutputLayer(int input_depth, int input_height, int input_width, int output_num);
        // Disable copy constructor
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer(OutputLayer&& other);

        inline int input_depth() const
        {
            return weights_view_.extent[0];
        }
        inline int input_height() const
        {
            return weights_view_.extent[1];
        }
        inline int input_width() const
        {
            return weights_view_.extent[2];
        }
        inline int output_num() const
        {
            return weights_view_.extent[3];
        }

        void SetLabel(int index);

        void ApplyBufferedUpdate(int buffer_size);

        void RandomizeParams(unsigned int seed);
    };

    // Contains a collection of neurons, which is 3-dimensional according to data layer.
    // So the model layer has 4-dimensional structure. Responsible for processing data layer
    // using neurons within and adjusting neuron weights during learning. 
    // Suppose dropout with fixed probability.
    class ConvolveLayer
    {
    private:
        // this vector is initialized before weight_view_
        std::vector<float> weights_;
        // bias for visible nodes, i.e. bottom nodes
        std::vector<float> vbias_;
        std::vector<float> hbias_;

    public:
        concurrency::array_view<float, 4>   weights_view_;
        concurrency::array<float, 4>        weights_delta_;

        // corresponding to the depth dimension
        concurrency::array_view<float>      vbias_view_;
        concurrency::array<float>           vbias_delta_;

        concurrency::array_view<float>      hbias_view_;
        concurrency::array<float>           hbias_delta_;

    public:
        ConvolveLayer(int num_neuron, int neuron_depth, int neuron_height, int neuron_width);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

        inline int neuron_num() const
        {
            return weights_view_.extent[0];
        }
        inline int neuron_depth() const
        {
            return weights_view_.extent[1];
        }
        inline int neuron_height() const
        {
            return weights_view_.extent[2];
        }
        inline int neuron_width() const
        {
            return weights_view_.extent[3];
        }

        void PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
            DataLayer& top_layer, bool top_switcher) const;

        void PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
            const OutputLayer& output_layer, bool output_switcher,
            DataLayer& top_layer, bool top_switcher) const;

        void PassDown(const DataLayer& top_layer, bool top_switcher,
            DataLayer& bottom_layer, bool bottom_switcher) const;

        void PassDown(const DataLayer& top_layer, bool top_switcher,
            DataLayer& bottom_layer, bool bottom_switcher,
            OutputLayer& output_layer, bool output_switcher) const;

        void Train(const DataLayer& bottom_layer, const DataLayer& top_layer,
            float learning_rate, bool buffered_update);

        void ApplyBufferedUpdate(int buffer_size);

        void RandomizeParams(unsigned int seed);

        bitmap_image GenerateImage() const;
    };

    // Pooling layer after convolvation, no params. 
    // Currently support max pooling, which is the most common pooling method.
    class PoolingLayer
    {
    public:
        int block_height_;
        int block_width_;

    public:
        PoolingLayer(int block_height, int block_width);
        // Disable copy constructor
        PoolingLayer(const PoolingLayer&) = delete;
        PoolingLayer(PoolingLayer&& other);

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

        void AddDataLayer(int depth, int height, int width, int seed = 0);
        void AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_height, int neuron_width,
            unsigned int rand_seed = 0);

        void PassUp(const std::vector<float>& data);
        void PassDown();

        float TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate, float dropout_prob);
        void TrainLayer(const std::vector<const std::vector<float>>& dataset,
            int layer_idx, int mini_batch_size, float learning_rate, float dropout_prob, int iter_count);

        void GenerateImages(const std::string& folder) const;

    private:
        std::vector<DataLayer> data_layers_;
        std::vector<ConvolveLayer> convolve_layers_;
        std::default_random_engine random_engine_;
    };
}

