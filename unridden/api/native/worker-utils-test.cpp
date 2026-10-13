#include "worker-utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

static bool close(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

static void check(bool condition) {
    if (!condition) throw std::runtime_error("worker utility assertion failed");
}

int main() {
    {
        const auto probabilities = unridden::stable_softmax({0.0, std::log(3.0)});
        check(close(probabilities.at(0), 0.25));
        check(close(probabilities.at(1), 0.75));
    }
    {
        const auto probabilities = unridden::stable_softmax(
            {10000.0, 9999.0, -10000.0});
        check(close(probabilities.at(0) + probabilities.at(1) + probabilities.at(2), 1.0));
        check(probabilities.at(0) > probabilities.at(1));
        check(probabilities.at(2) >= 0.0);
    }
    {
        const std::vector<double> vocabulary{0.0, 1.0, 2.0, -1.0};
        const auto summary = unridden::summarize_vocabulary(vocabulary, {1, 3});
        const double denominator = 1.0 + std::exp(1.0) + std::exp(2.0) + std::exp(-1.0);
        check(close(summary.allowed_mass, (std::exp(1.0) + std::exp(-1.0)) / denominator));
        check(summary.argmax_token_id == 2);
        check(summary.argmax_logit == 2.0);
    }
    {
        check(unridden::single_suffix_token(
            std::vector<int>{1, 2}, std::vector<int>{1, 2, 7}) == 7);
        bool rejected = false;
        try {
            (void) unridden::single_suffix_token(
                std::vector<int>{1, 2}, std::vector<int>{1, 9, 7});
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected);
    }
    {
        std::vector<float> rows{1.0f, 2.0f, 3.0f, 4.0f};
        unridden::apply_steering(
            rows, 2, {10.0f, 100.0f}, 0.5, unridden::steering_scope::all);
        check((rows == std::vector<float>{6.0f, 52.0f, 8.0f, 54.0f}));
    }
    {
        std::vector<float> rows{1.0f, 2.0f, 3.0f, 4.0f};
        unridden::apply_steering(
            rows, 2, {10.0f, 100.0f}, 1.0, unridden::steering_scope::last);
        check((rows == std::vector<float>{1.0f, 2.0f, 13.0f, 104.0f}));
        unridden::apply_steering(
            rows, 2, {10.0f, 100.0f}, 0.0, unridden::steering_scope::all);
        check((rows == std::vector<float>{1.0f, 2.0f, 13.0f, 104.0f}));
    }
    {
        // Every rejection leaves the rows untouched.
        const auto rejected = [](std::vector<float> rows, size_t width,
                                 const std::vector<float> & direction, double alpha) {
            const std::vector<float> before = rows;
            try {
                unridden::apply_steering(
                    rows, width, direction, alpha, unridden::steering_scope::all);
            } catch (const std::runtime_error &) {
                return rows == before;
            }
            return false;
        };
        check(rejected({1.0f, 2.0f}, 2, {1.0f}, 1.0));
        check(rejected({1.0f, 2.0f, 3.0f}, 2, {1.0f, 1.0f}, 1.0));
        check(rejected({}, 1, {1.0f}, 1.0));
        check(rejected({1.0f}, 1, {1.0f}, std::numeric_limits<double>::infinity()));
        check(rejected({1e38f}, 1, {1e38f}, 10.0));
    }
    {
        const auto chunks = unridden::prefill_chunks(513, 256);
        check((chunks == std::vector<size_t>{256, 256, 1}));
        bool rejected = false;
        try {
            (void) unridden::prefill_chunks(0, 256);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected);
    }
    {
        const std::vector<float> h{3.0f, 4.0f};
        const std::vector<float> gamma{2.0f, 0.5f};
        const auto normed = unridden::rms_norm(h.data(), gamma.data(), 2, 1e-6f);
        const double inv_rms = 1.0 / std::sqrt((9.0 + 16.0) / 2.0 + 1e-6);
        check(close(normed.at(0), 3.0 * inv_rms * 2.0, 1e-6));
        check(close(normed.at(1), 4.0 * inv_rms * 0.5, 1e-6));
        check(close(unridden::apply_softcap(0.0, 30.0), 0.0));
        check(close(unridden::apply_softcap(30.0, 30.0), 30.0 * std::tanh(1.0)));
    }
    {
        unridden::tuned_lens_model lens;
        lens.n_embd = 2;
        lens.rms_norm_eps = 1e-6f;
        lens.logit_softcap = 0.0;
        lens.label_token_ids = {10, 11, 12};
        lens.weight = {1.0f, 0.0f, 0.0f, 1.0f};
        lens.bias = {0.0f, 0.0f};
        lens.output_norm = {1.0f, 1.0f};
        lens.unembed = {2.0f, 0.0f, 0.0f, 2.0f, -1.0f, -1.0f};
        lens.gate.min_logit_margin = 0.4;
        lens.gate.min_entropy_concentration = 0.1;
        lens.gate.temperatures = {{2, 1.0}};
        lens.validate();

        const std::vector<float> h18{4.0f, 0.0f};
        std::vector<float> h30_pred;
        std::vector<float> normalized;
        lens.project_and_normalize(h18.data(), h30_pred, normalized);
        const auto logits = lens.score_labels(normalized);
        check(logits.size() == 3);
        check(logits.at(0) > logits.at(1));
        check(logits.at(1) > logits.at(2));

        check(lens.gate.passes({logits.at(0), logits.at(1)}));
        check(!lens.gate.passes({1.0, 0.99}));
    }
    return 0;
}
