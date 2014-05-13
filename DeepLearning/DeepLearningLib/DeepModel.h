#pragma once
#include <amp.h>
#include <vector>
#include <unordered_map>
#include <random>

// for random number generator on GPU
#include "amp_tinymt_rng.h"

// for bitmap generation
#include "bitmap_image.hpp"

namespace deep_learning_lib
{
    enum DataSlot
    {
        kCurrent, kNext, kTemp
    };

    // 4-dimensional data layer, cache the intermediate result in neural network
    // 3 dimension + time:
    //     _____________________
    //    /                    /|
    //   / height             / |
    //  /       width        /  |
    //  ---------------------   |
    //  |                   |   |
    //  | depth             |  /
    //  |                   | / 
    //  ---------------------/
    // depth dimension is orderless, representing the concept of unordered set.
    // time dimension is for short-term memory.
    class DataLayer
    {
    private:
        // these vectors are initialized before the corresponding array_views

        // internal storage for both value, expect_value, next_value, next_expect_value and short term memory.
        // simple unified storage because they all share the same structure. live on GPU directly.
        concurrency::array<float, 4> data_array_;
        int memory_num_;

        float active_prob_;
        std::vector<int>    active_;

        // if we don't forget/forgive, we cannot learn.
        const float kMemoryDecayRate = 0.99f;

    public:
        concurrency::array_view<float, 3>   value_view_;
        concurrency::array_view<float, 3>   expect_view_;
        concurrency::array_view<float, 3>   next_value_view_;
        concurrency::array_view<float, 3>   next_expect_view_;
        concurrency::array_view<float, 3>   temp_value_view_;
        concurrency::array_view<float, 3>   temp_expect_view_;

        // for dropout
        concurrency::array_view<int, 3>     active_view_;

        // short term memory view
        concurrency::array_view<float, 4>   memory_view_;

        tinymt_collection<3> rand_collection_;

    public:
        DataLayer(int memory_num, int depth, int height, int width, int seed = 0);
        // Disable copy constructor
        DataLayer(const DataLayer&) = delete;
        DataLayer(DataLayer&& other);

        void SetValue(const std::vector<float>& data);
        inline int memory_num() const
        {
            return memory_num_;
        }
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
        inline std::pair<concurrency::array_view<float, 3>, concurrency::array_view<float, 3>> operator[](const DataSlot data_slot) const
        {
            switch (data_slot)
            {
            case kCurrent:
                return std::make_pair(value_view_, expect_view_);
            case kNext:
                return std::make_pair(next_value_view_, next_expect_view_);
            case kTemp:
                return std::make_pair(temp_value_view_, temp_expect_view_);
            default:
                throw("DataLayer does not accept data slot type : " + std::to_string(data_slot));
            }
        }

        void Activate(float probability = 1.0f);

        float ReconstructionError() const;

        // Memorize current value if necessary. If a memory match is found, it will replace the current next value.
        // Data-driven, Nonparametric. Memorization is a kind of learning.
        // Return false if current value is already well learned thus not memorized.
        //bool Memorize();

        bitmap_image GenerateImage() const;
    };

    class ConvolveLayer;

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
        concurrency::array_view<float, 4>   weights_view_;

    public:
        OutputLayer(int output_num, int input_depth, int input_height, int input_width);
        // Disable copy constructor
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer(OutputLayer&& other);

        inline int output_num() const
        {
            return weights_view_.extent[0];
        }
        inline int input_depth() const
        {
            return weights_view_.extent[1];
        }
        inline int input_height() const
        {
            return weights_view_.extent[2];
        }
        inline int input_width() const
        {
            return weights_view_.extent[3];
        }
        inline concurrency::array_view<float> operator[](const DataSlot data_slot) const
        {
            switch (data_slot)
            {
            case kCurrent:
                return outputs_view_;
            case kNext:
                return next_outputs_view_;
            default:
                throw("OutputLayer does not accept data slot type : " + std::to_string(data_slot));
            }
        }

        void SetLabel(const int label);

        void RandomizeParams(unsigned int seed);

        int PredictLabel(
            const DataLayer& bottom_layer, bool bottom_switcher,
            DataLayer& top_layer, bool top_switcher,
            const ConvolveLayer& conv_layer, const float dropout_prob);

        void PassDown(const DataLayer& top_layer, bool top_switcher, bool output_switcher);

        bitmap_image GenerateImage() const;
    };

    // Contains a collection of neurons, which is 4-dimensional according to data layer.
    // So the model layer has 5-dimensional structure. 
    // 1. Responsible for processing data layer using neurons within and adjusting neuron weights during learning.
    // 2. Responsible for long term memory logic.
    // 3. Suppose dropout with fixed probability.
    class ConvolveLayer
    {
    private:
        // traditional neuron weights and longterm memory weights are both contained in this vector
        // since they share the same structure
        std::vector<float> weights_;

        // bias for visible nodes, i.e. bottom nodes
        std::vector<float> vbias_;
        std::vector<float> hbias_;

        int longterm_memory_num_;
        int shortterm_memory_num_;

    public:
        // long term memory + neuron
        concurrency::array_view<float, 5>   neurons_with_longterm_memory_view_;
        // long term memory view
        concurrency::array_view<float, 5>   longterm_memory_view_;
        // traditional neurons weight view, containing weights for short-term memories in bottom layer
        concurrency::array_view<float, 5>   neurons_view_;
        // neurons weight view only for value layer in bottom layer, no short memory part, for training
        concurrency::array_view<float, 5>   neurons_no_shortterm_memory_view_;

        // corresponding to the depth dimension
        concurrency::array_view<float>      vbias_view_;
        concurrency::array_view<float>      hbias_view_;

    public:
        ConvolveLayer(int longterm_memory_num, int neuron_num,
            int shortterm_memory_num, int neuron_depth, int neuron_height, int neuron_width);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

        inline int longterm_memory_num() const
        {
            return longterm_memory_num_;
        }
        inline int shortterm_memory_num() const
        {
            return shortterm_memory_num_;
        }
        inline int neuron_num() const
        {
            return neurons_view_.extent[0];
        }
        inline int neuron_depth() const
        {
            return neurons_view_.extent[2];
        }
        inline int neuron_height() const
        {
            return neurons_view_.extent[3];
        }
        inline int neuron_width() const
        {
            return neurons_view_.extent[4];
        }

        void PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
            DataLayer& top_layer, DataSlot top_slot,
            const OutputLayer* output_layer = nullptr, DataSlot output_slot = kCurrent) const;

        void PassDown(const DataLayer& top_layer, DataSlot top_slot,
            DataLayer& bottom_layer, DataSlot bottom_slot,
            OutputLayer* output_layer = nullptr, DataSlot output_slot = kCurrent) const;

        // Not all long-term memory activations are helpful, let's filter these harmful memories.
        void SuppressMemory(DataLayer& top_layer, DataSlot top_slot,
            const DataLayer& bottom_layer, DataSlot bottom_slot) const;

        // generative or discriminative training
        void Train(const DataLayer& bottom_layer, const DataLayer& top_layer, float learning_rate,
            OutputLayer* output_layer = nullptr, bool discriminative_training = false);

        void RandomizeParams(unsigned int seed);

        bitmap_image GenerateImage() const;
    };

    // Pooling layer after convolution, no params. 
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
        explicit DeepModel(unsigned int model_seed = 0);
        // Disable copy constructor
        DeepModel(const DeepModel&) = delete;

        void AddDataLayer(int memory_num, int depth, int height, int width);
        void AddConvolveLayer(int memory_num, int neuron_num, int neuron_depth, int neuron_height, int neuron_width);
        void AddOutputLayer(int data_layer_idx, int output_num);

        void PassUp(const std::vector<float>& data);
        void PassDown();

        float TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate, float dropout_prob,
            const int label = -1, bool discriminative_training = false);

        int PredictLabel(const std::vector<float>& data, const int layer_idx, const float dropout_prob);

        float Evaluate(const std::vector<const std::vector<float>>& dataset, const std::vector<const int>& labels,
            int layer_idx, const float dropout_prob);

        void GenerateImages(const std::string& folder) const;

    private:
        std::vector<DataLayer> data_layers_;
        std::vector<ConvolveLayer> convolve_layers_;
        std::unordered_map<int, OutputLayer> output_layers_;
        std::default_random_engine random_engine_;
    };
}

