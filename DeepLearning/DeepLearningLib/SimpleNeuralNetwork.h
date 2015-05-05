#pragma once
#include <amp.h>
#include <vector>
#include <string>

// for random number generator on GPU, maximum capacity = 65535 generators
#include "amp_tinymt_rng.h"

namespace deep_learning_lib
{
    class SimpleNN
    {
    public:
        explicit SimpleNN(int bottom_length, int top_length, unsigned int seed = 0);
        SimpleNN(const SimpleNN&) = delete;

        double Feed(const std::vector<int>& input_data, double data_weight = 0.0);

        void Dump(const std::string& folder, const std::string& tag = "") const;

    private:
        void RandomizeParams(unsigned int seed);

        void PassUp();
        void PassDown();

        void Learn(double data_weight = 0.0);

        double CalcReconError() const;

        void Observe();

    private:
        int bottom_length_;
        int top_length_;

        // model parameters to learn
        // [top, bottom]
        concurrency::array_view<double, 2> neuron_weights_;
        // indicates bottom neuron innate activation expect
        concurrency::array_view<double> bottom_biases_;
        // indicates top neuron activation frequency
        concurrency::array_view<double> top_biases_;

        // current data
        concurrency::array_view<double> top_expects_;
        concurrency::array_view<int> top_values_;

        concurrency::array_view<int> bottom_values_;
        concurrency::array_view<double> bottom_recon_expects_;
        concurrency::array_view<int> bottom_clusters_;

        concurrency::array_view<double, 2> bottom_up_messages_;
        concurrency::array_view<double> top_energies_;

        //
        tinymt_collection<1> top_rand_;

        const int kInferenceCount = 5;
        const double kNeuronTolerance = 1.0;
    };
}