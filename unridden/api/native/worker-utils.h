#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace unridden {

struct vocabulary_summary {
    double allowed_mass;
    int argmax_token_id;
    double argmax_logit;
};

inline std::vector<double> stable_softmax(const std::vector<double> & logits) {
    if (logits.size() < 2) throw std::runtime_error("Need at least two label logits");
    for (const double value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite label logit");
    }
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double total = 0.0;
    for (const double value : logits) {
        probabilities.push_back(std::exp(value - maximum));
        total += probabilities.back();
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error("Invalid label normalization");
    }
    for (double & value : probabilities) value /= total;
    const double check = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (!std::isfinite(check) || std::abs(check - 1.0) > 1e-12) {
        throw std::runtime_error("Label probabilities are not normalized");
    }
    return probabilities;
}

inline vocabulary_summary summarize_vocabulary(
        const std::vector<double> & logits,
        const std::vector<int> & allowed_token_ids) {
    if (logits.empty()) throw std::runtime_error("Empty vocabulary logits");
    const auto argmax = std::max_element(logits.begin(), logits.end());
    for (const double value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite vocabulary logit");
    }
    const double maximum = *argmax;
    double denominator = 0.0;
    for (const double value : logits) denominator += std::exp(value - maximum);
    double numerator = 0.0;
    for (const int token_id : allowed_token_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= logits.size()) {
            throw std::runtime_error("Allowed token is outside vocabulary");
        }
        numerator += std::exp(logits.at(static_cast<size_t>(token_id)) - maximum);
    }
    const double mass = numerator / denominator;
    if (!std::isfinite(mass) || mass < 0.0 || mass > 1.0) {
        throw std::runtime_error("Invalid allowed-label mass");
    }
    return {
        mass,
        static_cast<int>(std::distance(logits.begin(), argmax)),
        *argmax,
    };
}

template <typename Token>
Token single_suffix_token(
        const std::vector<Token> & prompt,
        const std::vector<Token> & extended) {
    if (extended.size() != prompt.size() + 1 ||
            !std::equal(prompt.begin(), prompt.end(), extended.begin())) {
        throw std::runtime_error("Label is not one suffix token for the exact prompt");
    }
    return extended.back();
}

enum class steering_scope { all, last };

// Adds `alpha * direction` to every row of a [rows, width] residual matrix, or
// to its final row only, in place. The arithmetic is done on a copy and only
// committed once every result is finite, so a rejected steer leaves the rows
// exactly as they were.
inline void apply_steering(
        std::vector<float> & rows,
        size_t width,
        const std::vector<float> & direction,
        double alpha,
        steering_scope scope) {
    if (width == 0 || rows.empty() || rows.size() % width != 0) {
        throw std::runtime_error("Steering rows are not a matrix of the given width");
    }
    if (direction.size() != width) {
        throw std::runtime_error("Steering direction width differs from the rows");
    }
    if (!std::isfinite(alpha)) throw std::runtime_error("Steering alpha is nonfinite");
    for (const float value : direction) {
        if (!std::isfinite(value)) throw std::runtime_error("Steering direction is nonfinite");
    }
    std::vector<float> steered = rows;
    const size_t row_count = rows.size() / width;
    const size_t first = scope == steering_scope::last ? row_count - 1 : 0;
    for (size_t row = first; row < row_count; ++row) {
        for (size_t index = 0; index < width; ++index) {
            const float value = static_cast<float>(
                static_cast<double>(steered[row * width + index])
                + alpha * static_cast<double>(direction[index]));
            if (!std::isfinite(value)) {
                throw std::runtime_error("Steering produced a nonfinite residual");
            }
            steered[row * width + index] = value;
        }
    }
    rows = std::move(steered);
}

inline std::vector<size_t> prefill_chunks(size_t token_count, size_t batch_size) {
    if (token_count == 0 || batch_size == 0) {
        throw std::runtime_error("Token and batch counts must be positive");
    }
    std::vector<size_t> chunks;
    for (size_t offset = 0; offset < token_count; offset += batch_size) {
        chunks.push_back(std::min(batch_size, token_count - offset));
    }
    return chunks;
}

inline double logit_margin(const std::vector<double> & logits) {
    if (logits.size() < 2) throw std::runtime_error("Need at least two label logits");
    for (const double value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite label logit");
    }
    std::vector<double> ordered = logits;
    std::partial_sort(ordered.begin(), ordered.begin() + 2, ordered.end(), std::greater<double>());
    return std::max(0.0, ordered[0] - ordered[1]);
}

inline double entropy_concentration(const std::vector<double> & probabilities) {
    if (probabilities.size() < 2) {
        throw std::runtime_error("Need at least two label probabilities");
    }
    double entropy = 0.0;
    for (const double item : probabilities) {
        if (!std::isfinite(item) || item < 0.0 || item > 1.0) {
            throw std::runtime_error("Invalid label probability");
        }
        if (item > 0.0) entropy -= item * std::log(item);
    }
    const double scale = std::log(static_cast<double>(probabilities.size()));
    if (scale <= 0.0) return 1.0;
    const double normalized = std::min(1.0, std::max(0.0, entropy / scale));
    return std::min(1.0, std::max(0.0, 1.0 - normalized));
}

inline std::vector<float> rms_norm(
        const float * input,
        const float * weight,
        size_t n_embd,
        float eps) {
    if (n_embd == 0 || !std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error("Invalid RMSNorm parameters");
    }
    double sum_sq = 0.0;
    for (size_t index = 0; index < n_embd; ++index) {
        if (!std::isfinite(input[index]) || !std::isfinite(weight[index])) {
            throw std::runtime_error("Nonfinite RMSNorm input or weight");
        }
        const double value = static_cast<double>(input[index]);
        sum_sq += value * value;
    }
    const double scale = 1.0 / std::sqrt(sum_sq / static_cast<double>(n_embd) + static_cast<double>(eps));
    std::vector<float> output(n_embd);
    for (size_t index = 0; index < n_embd; ++index) {
        const float value = static_cast<float>(
            static_cast<double>(input[index]) * scale * static_cast<double>(weight[index]));
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite RMSNorm output");
        output[index] = value;
    }
    return output;
}

inline double apply_softcap(double logit, double softcap) {
    if (!std::isfinite(logit) || !std::isfinite(softcap) || softcap < 0.0) {
        throw std::runtime_error("Invalid softcap parameters");
    }
    if (softcap == 0.0) return logit;
    return softcap * std::tanh(logit / softcap);
}

struct tuned_lens_gate {
    bool enabled = true;
    double min_logit_margin = 0.0;
    double min_entropy_concentration = 0.0;
    std::vector<std::pair<size_t, double>> temperatures;

    double temperature_for(size_t option_count) const {
        for (const auto & [count, temperature] : temperatures) {
            if (count == option_count) return temperature;
        }
        return 1.0;
    }

    bool passes(const std::vector<double> & label_logits) const {
        (void) stable_softmax(label_logits);
        if (!enabled) return true;
        const double margin = logit_margin(label_logits);
        const double temperature = temperature_for(label_logits.size());
        if (!std::isfinite(temperature) || temperature <= 0.0) {
            throw std::runtime_error("Invalid gate temperature");
        }
        std::vector<double> scaled;
        scaled.reserve(label_logits.size());
        for (const double value : label_logits) scaled.push_back(value / temperature);
        const auto probabilities = stable_softmax(scaled);
        const double concentration = entropy_concentration(probabilities);
        return margin >= min_logit_margin && concentration >= min_entropy_concentration;
    }
};

struct tuned_lens_model {
    size_t n_embd = 0;
    float rms_norm_eps = 1e-6f;
    double logit_softcap = 30.0;
    std::vector<int> label_token_ids;
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> output_norm;
    std::vector<float> unembed;
    tuned_lens_gate gate;

    void validate() const {
        if (n_embd == 0 || label_token_ids.size() < 2) {
            throw std::runtime_error("Invalid Tuned Lens dimensions");
        }
        if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0f) {
            throw std::runtime_error("Invalid Tuned Lens rms_norm_eps");
        }
        if (!std::isfinite(logit_softcap) || logit_softcap < 0.0) {
            throw std::runtime_error("Invalid Tuned Lens logit_softcap");
        }
        if (weight.size() != n_embd * n_embd ||
                bias.size() != n_embd ||
                output_norm.size() != n_embd ||
                unembed.size() != label_token_ids.size() * n_embd) {
            throw std::runtime_error("Tuned Lens tensor size mismatch");
        }
        for (const float value : weight) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Tuned Lens weight");
        }
        for (const float value : bias) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Tuned Lens bias");
        }
        for (const float value : output_norm) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Tuned Lens output_norm");
        }
        for (const float value : unembed) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Tuned Lens unembed");
        }
        if (!std::isfinite(gate.min_logit_margin) || gate.min_logit_margin < 0.0 ||
                !std::isfinite(gate.min_entropy_concentration) ||
                gate.min_entropy_concentration < 0.0 ||
                gate.min_entropy_concentration > 1.0) {
            throw std::runtime_error("Invalid Tuned Lens gate thresholds");
        }
        for (const auto & [count, temperature] : gate.temperatures) {
            if (count < 2 || !std::isfinite(temperature) || temperature <= 0.0) {
                throw std::runtime_error("Invalid Tuned Lens gate temperature");
            }
        }
    }

    void project_and_normalize(
            const float * h18_last,
            std::vector<float> & h30_pred,
            std::vector<float> & normalized) const {
        if (!h18_last) throw std::runtime_error("Missing H18 input");
        for (size_t index = 0; index < n_embd; ++index) {
            if (!std::isfinite(h18_last[index])) {
                throw std::runtime_error("Nonfinite H18 input");
            }
        }
        h30_pred.resize(n_embd);
        for (size_t row = 0; row < n_embd; ++row) {
            double accum = static_cast<double>(bias[row]);
            const float * weight_row = weight.data() + row * n_embd;
            for (size_t col = 0; col < n_embd; ++col) {
                accum += static_cast<double>(weight_row[col]) * static_cast<double>(h18_last[col]);
            }
            const float projected = static_cast<float>(accum);
            if (!std::isfinite(projected)) {
                throw std::runtime_error("Nonfinite Tuned Lens projection");
            }
            h30_pred[row] = projected;
        }
        normalized = rms_norm(h30_pred.data(), output_norm.data(), n_embd, rms_norm_eps);
    }

    std::vector<double> score_labels(const std::vector<float> & normalized) const {
        if (normalized.size() != n_embd) {
            throw std::runtime_error("Normalized vector width mismatch");
        }
        const size_t label_count = label_token_ids.size();
        std::vector<double> logits(label_count);
        for (size_t label = 0; label < label_count; ++label) {
            double dot = 0.0;
            const float * row = unembed.data() + label * n_embd;
            for (size_t col = 0; col < n_embd; ++col) {
                dot += static_cast<double>(row[col]) * static_cast<double>(normalized[col]);
            }
            logits[label] = apply_softcap(dot, logit_softcap);
        }
        return logits;
    }
};

}  // namespace unridden
