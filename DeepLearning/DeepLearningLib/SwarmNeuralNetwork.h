#pragma once
#include <amp.h>
#include <vector>
#include <string>

// for random number generator on GPU, maximum capacity = 65535 generators
#include "amp_tinymt_rng.h"

namespace deep_learning_lib
{
    class SwarmNN
    {
    public:
        explicit SwarmNN(int bottom_length, int top_length, unsigned int seed = 0);
        SwarmNN(const SwarmNN&) = delete;

        double Feed(const std::vector<int>& input_data, double data_weight = 0);

        void Dump(const std::string& folder, const std::string& tag = "") const;

    private:
        void RandomizeParams(unsigned int seed);

        void PassUp(double data_weight = 0);
        void PassDown(double data_weight = 0);

        double CalcReconError() const;

    private:
        int bottom_length_;
        int top_length_;

        // model parameters to learn
        // [top, bottom]
        concurrency::array_view<double, 2> neuron_weights_;
        // indicates bottom neuron innate activation expect
        concurrency::array_view<double> bottom_biases_;
        // indicates top neuron activation frequence
        concurrency::array_view<double> top_biases_;

        // current data
        concurrency::array_view<int> bottom_values_;
        concurrency::array_view<int> top_values_;
        concurrency::array_view<int> bottom_recon_values_;
        concurrency::array_view<double> bottom_recon_raw_weights_;

        // for debug
        concurrency::array_view<double> top_expects_;

        //
        tinymt_collection<1> bottom_rand_;
        tinymt_collection<1> top_rand_;

        const double kPowerLawFactor = 0.5;
        const double kPowerLawCutoff = 0.05;
        const int kInferenceCount = 5;
    };
}