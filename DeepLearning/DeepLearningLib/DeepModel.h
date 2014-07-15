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
    enum class DataSlot
    {
        kCurrent,
        kNext,
        kTemp
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
        float active_prob_; // for dropout
        int shortterm_memory_num_;

    public:
        // value = longterm memory + neuron
        concurrency::array_view<float, 3> value_view_;
        concurrency::array_view<float, 3> expect_view_;
        concurrency::array_view<float, 3> next_value_view_;
        concurrency::array_view<float, 3> next_expect_view_;
        concurrency::array_view<float, 3> temp_value_view_;
        concurrency::array_view<float, 3> temp_expect_view_;

        // for evaluation, store the bottom up sum of neuron weights
        concurrency::array_view<float, 3> raw_weight_view_;

        // for dropout
        concurrency::array_view<int, 3> active_view_;

        // short term memory view
        concurrency::array_view<float, 4> shortterm_memory_view_;
        // index into shortterm memory in temporal order.
        concurrency::array_view<int, 1> shortterm_memory_index_view_;

        tinymt_collection<3> rand_collection_;

    public:
        DataLayer(int shortterm_memory_num, int depth, int height, int width, int seed = 0);
        // Disable copy constructor
        DataLayer(const DataLayer&) = delete;
        DataLayer(DataLayer&& other);

        void SetValue(const std::vector<float>& data);

        inline int shortterm_memory_num() const
        {
            return shortterm_memory_num_;
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

        inline std::pair<concurrency::array_view<float, 3>, concurrency::array_view<float, 3>>
            operator[](const DataSlot data_slot) const
        {
            switch (data_slot)
            {
            case DataSlot::kCurrent:
                return std::make_pair(value_view_, expect_view_);
            case DataSlot::kNext:
                return std::make_pair(next_value_view_, next_expect_view_);
            case DataSlot::kTemp:
                return std::make_pair(temp_value_view_, temp_expect_view_);
            default:
                throw("Invalid data slot type for data layer.");
            }
        }

        void Activate(float active_prob = 1.0f);

        // store current value into shortterm memory.
        void Memorize();

        float ReconstructionError() const;

        bitmap_image GenerateImage() const;
    };

    class ConvolveLayer;

    // Currently support 1-of-N classifier output.
    // It contains both data and weight parameters.
    // Support both discriminative and generative training.
    class OutputLayer
    {
    private:
        std::vector<float> bias_;
        std::vector<float> weights_;

    public:
        concurrency::array_view<float> outputs_view_;
        concurrency::array_view<float> next_outputs_view_;
        concurrency::array_view<float> temp_outputs_view_;

        concurrency::array_view<float> bias_view_;
        concurrency::array_view<float, 4> weights_view_;

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
            case DataSlot::kCurrent:
                return outputs_view_;
            case DataSlot::kNext:
                return next_outputs_view_;
            case DataSlot::kTemp:
                return temp_outputs_view_;
            default:
                throw("Invalid data slot type for output layer.");
            }
        }

        void SetLabel(const int label);

        void RandomizeParams(unsigned int seed);

        int PredictLabel(DataLayer& bottom_layer, DataSlot bottom_slot, DataLayer& top_layer, DataSlot top_slot,
            const ConvolveLayer& conv_layer, const float dropout_prob);

        void PassDown(const DataLayer& top_layer, DataSlot top_slot, DataSlot output_slot);

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
        std::vector<float> neuron_weights_;
        std::vector<float> longterm_memory_weights_;

        // store how good is the reconstruction of model against true data at each positions
        // longterm memory below this threshold is suppressed, so it serves as a sparse prior
        concurrency::array_view<float, 2> longterm_memory_affinity_prior_view_;

        // record the index of longterm memory with max affinity at this position on top layer
        Concurrency::array_view<int, 2> longterm_memory_max_affinity_index_view_;

        // longterm memory activation info when passing up
        // it's not stored in data layer because long term memory is transparent to data layer
        // [longterm_memory_idx, height_idx, width_idx]
        concurrency::array_view<float, 3> longterm_memory_affinity_view_;

        // the cumulative gain for each longterm memory.
        // the min of this will be replaced by newly encountered memory
        concurrency::array_view<float> longterm_memory_gain_view_;
        // if we cannot forget, we cannot learn.
        // exponential decay used in cumulative longterm memory gain.
        // maybe this should be adaptive instead of constant.
        const float kLongtermMemoryDecay = 0.99f;

        // bias for visible nodes, i.e. bottom nodes
        std::vector<float> vbias_;
        std::vector<float> hbias_;

        int longterm_memory_num_;
        int longterm_memory_depth_;

    public:
        // neurons weight view [neuron_idx, neuron_depth, neuron_height, neuron_width]
        concurrency::array_view<float, 4> neuron_weights_view_;
        // longterm memory view [longterm_memory_idx, neuron_depth, neuron_height, neuron_width]
        concurrency::array_view<float, 4> longterm_memory_weights_view_;

        // corresponding to the depth dimension
        concurrency::array_view<float> vbias_view_;
        concurrency::array_view<float> hbias_view_;

    public:
        ConvolveLayer(int longterm_memory_num, int longterm_memory_depth,
            int neuron_num, int neuron_depth, int neuron_height, int neuron_width);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

        inline int longterm_memory_num() const
        {
            return longterm_memory_num_;
        }

        inline int longterm_memory_depth() const
        {
            return longterm_memory_depth_;
        }

        inline int neuron_num() const
        {
            return neuron_weights_view_.extent[0];
        }

        inline int neuron_depth() const
        {
            return neuron_weights_view_.extent[1];
        }

        inline int neuron_height() const
        {
            return neuron_weights_view_.extent[2];
        }

        inline int neuron_width() const
        {
            return neuron_weights_view_.extent[3];
        }

        void PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
            DataLayer& top_layer, DataSlot top_slot,
            const OutputLayer* output_layer = nullptr, DataSlot output_slot = DataSlot::kCurrent) const;

        void PassDown(const DataLayer& top_layer, DataSlot top_slot,
            DataLayer& bottom_layer, DataSlot bottom_slot,
            OutputLayer* output_layer = nullptr, DataSlot output_slot = DataSlot::kCurrent) const;

        // Activate long-term memory nodes in top layer.
        void ActivateMemory(DataLayer& top_layer, DataSlot top_slot,
            const DataLayer& bottom_layer, DataSlot bottom_data_slot, DataSlot bottom_model_slot) const;

        // generative or discriminative training
        void Train(const DataLayer& bottom_layer, const DataLayer& top_layer, float learning_rate,
            OutputLayer* output_layer = nullptr, bool discriminative_training = false);

        bool FitLongtermMemory(const DataLayer& top_layer);

        void RandomizeParams(unsigned int seed);

        bitmap_image GenerateImage() const;
    };

    // Pooling layer after convolution, no params. 
    // Currently support max pooling, which is the most common pooling method.
    class PoolingLayer
    {
    private:
        int block_height_;
        int block_width_;

    public:
        PoolingLayer(int block_height, int block_width);
        // Disable copy constructor
        PoolingLayer(const PoolingLayer&) = delete;
        PoolingLayer(PoolingLayer&& other);

        inline int block_height() const
        {
            return block_height_;
        }

        inline int block_width() const
        {
            return block_width_;
        }

        void PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
            DataLayer& top_layer, DataSlot top_slot) const;

        void PassDown(const DataLayer& top_layer, DataSlot top_slot,
            DataLayer& bottom_layer, DataSlot bottom_slot) const;
    };

    class DeepModel
    {
    public:
        explicit DeepModel(unsigned int model_seed = 0);
        // Disable copy constructor
        DeepModel(const DeepModel&) = delete;

        // only used for adding the first data layer
        void AddDataLayer(int depth, int height, int width, int shortterm_memory_num = 0);
        // deduce the parameters from the convolve layer below
        void AddDataLayer(int shortterm_memory_num = 0);

        // deduce the parameters from the data layer below
        void AddConvolveLayer(int neuron_num, int neuron_height, int neuron_width, int longterm_memory_num = 0);

        void AddOutputLayer(int output_num);

        void PassUp(const std::vector<float>& data);
        void PassDown();

        float TrainLayer(const std::vector<float>& data, int layer_idx,
            float learning_rate, float dropout_prob,
            const int label = -1, bool discriminative_training = false);

        int PredictLabel(const std::vector<float>& data, const int layer_idx, const float dropout_prob);

        float Evaluate(const std::vector<const std::vector<float>>& dataset, const std::vector<const int>& labels,
            int layer_idx, const float dropout_prob);

        void GenerateImages(const std::string& folder) const;

    private:
        enum class LayerType
        {
            kDataLayer,
            kConvolveLayer,
            kPoolingLayer,
            kOutputLayer
        };

        // layer type -> the index into each vector of that type
        // common pattern: data layer <-> convolve layer <-> data layer <-> pooling layer <-> data layer <-> convolve layer ...
        std::vector<std::pair<LayerType, size_t>> layer_stack_;

        std::vector<DataLayer> data_layers_;
        std::vector<ConvolveLayer> convolve_layers_;
        std::vector<PoolingLayer> pooling_layers;

        // data layer index => output layer. 
        // please note that output layer is attached to data layer only. 
        // the data flow differs from other layer types, so I exclude it from the layer stack.
        std::unordered_map<size_t, OutputLayer> output_layers_;

        std::default_random_engine random_engine_;
    };
}