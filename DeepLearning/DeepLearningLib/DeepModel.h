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
    enum class DataSlotType
    {
        kCurrent,
        kNext,
        kTemp,
        kInvalid
    };

    class ConvolveLayer;

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
        int shortterm_memory_num_ = 0;

    public:
        struct DataSlot
        {
            concurrency::array_view<double, 3> values_view_;
            concurrency::array_view<double, 3> expects_view_;
            // raw weights before neuron activation
            // [depth_idx = neuron_idx, height_idx, width_idx]
            concurrency::array_view<double, 3> raw_weights_view_;

            explicit DataSlot(int depth, int height, int width);
        };

    public:
        DataSlot cur_data_slot_;
        DataSlot next_data_slot_;
        DataSlot tmp_data_slot_;

        // short term memory view
        concurrency::array_view<double, 4> shortterm_memories_view_;
        // index into shortterm memory in temporal order.
        concurrency::array_view<int, 1> shortterm_memory_index_view_;

        tinymt_collection<3> rand_collection_;

    public:
        explicit DataLayer(int shortterm_memory_num, int depth, int height, int width, int seed = 0);

        inline int shortterm_memory_num() const
        {
            return shortterm_memory_num_;
        }

        inline int depth() const
        {
            return cur_data_slot_.values_view_.extent[0];
        }

        inline int height() const
        {
            return cur_data_slot_.values_view_.extent[1];
        }

        inline int width() const
        {
            return cur_data_slot_.values_view_.extent[2];
        }

        inline const DataSlot& operator[](const DataSlotType data_slot_type) const
        {
            switch (data_slot_type)
            {
            case DataSlotType::kCurrent:
                return cur_data_slot_;
            case DataSlotType::kNext:
                return next_data_slot_;
            case DataSlotType::kTemp:
                return tmp_data_slot_;
            default:
                throw("Invalid data slot type for data layer.");
            }
        }

        void SetValue(const std::vector<double>& data);

        // store current value into shortterm memory.
        void Memorize();

        float ReconstructionError(DataSlotType slot_type) const;

        bitmap_image GenerateImage() const;
    };

    // Currently support 1-of-N classifier output.
    // It contains both data and weight parameters.
    // Support both discriminative and generative training.
    class OutputLayer
    {
    private:
        std::vector<double> bias_;
        std::vector<double> neuron_weights_;

    public:
        struct DataSlot
        {
            concurrency::array_view<double> outputs_view_;
            concurrency::array_view<double> raw_weights_view_;

            explicit DataSlot(int output_num);
        };

    public:
        DataSlot cur_data_slot_;
        DataSlot next_data_slot_;
        DataSlot tmp_data_slot_;

        concurrency::array_view<double> bias_view_;
        concurrency::array_view<double, 4> neuron_weights_view_;

    public:
        explicit OutputLayer(int output_num, int input_depth, int input_height, int input_width);
        // Disable copy constructor
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer(OutputLayer&& other);

        inline int output_num() const
        {
            return neuron_weights_view_.extent[0];
        }

        inline int input_depth() const
        {
            return neuron_weights_view_.extent[1];
        }

        inline int input_height() const
        {
            return neuron_weights_view_.extent[2];
        }

        inline int input_width() const
        {
            return neuron_weights_view_.extent[3];
        }

        inline const DataSlot& operator[](const DataSlotType data_slot_type) const
        {
            switch (data_slot_type)
            {
            case DataSlotType::kCurrent:
                return cur_data_slot_;
            case DataSlotType::kNext:
                return next_data_slot_;
            case DataSlotType::kTemp:
                return tmp_data_slot_;
            default:
                throw("Invalid data slot type for output layer.");
            }
        }

        void SetLabel(const int label);

        void RandomizeParams(unsigned int seed);

        int PredictLabel(DataLayer& bottom_layer, DataLayer& top_layer, const ConvolveLayer& conv_layer);

        void PassDown(const DataLayer& top_layer, DataSlotType top_slot_type, DataSlotType output_slot_type);

        bitmap_image GenerateImage() const;
    };

    // Contains a collection of neurons, which is 4-dimensional according to data layer.
    // So the model layer has 5-dimensional structure. 
    // 1. Responsible for processing data layer using neurons within and adjusting neuron weights during learning.
    // 2. Responsible for long term memory logic.
    class ConvolveLayer
    {
    private:
        // parameters we need to learn
        std::vector<double> neuron_weights_;
        
        // bias for visible nodes, i.e. bottom nodes
        std::vector<double> vbias_;
        std::vector<double> hbias_;

        const int kPassUpIteration = 10;

    public:
        // neurons weight view [neuron_idx, neuron_depth, neuron_height, neuron_width]
        concurrency::array_view<double, 4> neuron_weights_view_;

        // corresponding to the depth dimension
        concurrency::array_view<double> vbias_view_;
        concurrency::array_view<double> hbias_view_;

    public:
        explicit ConvolveLayer(int neuron_num, int neuron_depth, int neuron_height, int neuron_width);
        // Disable copy constructor
        ConvolveLayer(const ConvolveLayer&) = delete;
        ConvolveLayer(ConvolveLayer&& other);

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

        void PassUp(DataLayer& bottom_layer, DataSlotType bottom_slot_type,
            DataLayer& top_layer, DataSlotType top_slot_type,
            OutputLayer* output_layer = nullptr, DataSlotType output_slot_type = DataSlotType::kInvalid) const;

        void PassDown(const DataLayer& top_layer, DataSlotType top_slot_type,
            DataLayer& bottom_layer, DataSlotType bottom_slot_type,
            OutputLayer* output_layer = nullptr, DataSlotType output_slot_type = DataSlotType::kInvalid) const;

        // generative or discriminative training
        void Train(const DataLayer& bottom_layer, const DataLayer& top_layer, double learning_rate,
            OutputLayer* output_layer = nullptr, bool discriminative_training = false);

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

        void PassUp(const DataLayer& bottom_layer, DataSlotType bottom_slot_type,
            DataLayer& top_layer, DataSlotType top_slot_type) const;

        void PassDown(const DataLayer& top_layer, DataSlotType top_slot_type,
            DataLayer& bottom_layer, DataSlotType bottom_slot_type) const;
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
        void AddConvolveLayer(int neuron_num, int neuron_height, int neuron_width);

        void AddOutputLayer(int output_num);

        void PassUp(const std::vector<double>& data);
        void PassDown();

        double TrainLayer(const std::vector<double>& data, int layer_idx,
            double learning_rate, const int label = -1, bool discriminative_training = false);

        int PredictLabel(const std::vector<double>& data, const int layer_idx);

        std::pair<double, std::vector<std::tuple<int, int, int>>> Evaluate(const std::vector<const std::vector<double>>& dataset,
            const std::vector<const int>& labels, int layer_idx);

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