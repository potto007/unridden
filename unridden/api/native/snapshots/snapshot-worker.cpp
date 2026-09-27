// Unridden state-snapshot worker, profile split18-30-v1.
//
// One Gemma 4 model, two llama.cpp contexts that share its weights: `lower`
// executes and owns the K/V of blocks 0-17, `upper` of blocks 18-29 plus the
// final norm and head. The residual entering block 18 (H18) is the only thing
// that crosses between them. It needs the gemma4-layer-range patch; the
// protocol is docs/superpowers/specs/2026-09-22-state-snapshots-protocol.md.
//
// Snapshots are immutable host-side records. Every branch restores (or finds
// still resident) its parent's exact per-range sequence state, appends its own
// suffix, reads out, and trims back, so siblings never see each other.
//
// Profile full-v1 (ADR 0009) keeps the same protocol with one context over
// the whole model, the stock graph, so it serves any Gemma 4, E4B included.
// That context is the protocol's `lower` range, the only checkpoint is the
// model's block count, and the `h30` slot holds the residual after the final
// block.

#include "chat.h"
#include "llama.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include "worker-utils.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {

constexpr size_t MAX_PROTOCOL_BYTES = 4 * 1024 * 1024;
constexpr const char * PROTOCOL = "unridden-snapshot-v1";
constexpr const char * PROFILE = "split18-30-v1";
constexpr const char * FULL_PROFILE = "full-v1";
constexpr const char * MODEL_ID = "local-gemma-unridden-v1";
constexpr const char * CONTEXT_PROMPT_VERSION = "unridden-gemma-context-v1";
constexpr const char * LABEL_CANDIDATES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr int EXPECTED_LAYERS = 30;
constexpr int SPLIT_BLOCK = 18;
constexpr size_t MAX_VECTOR_ROWS = 64;
constexpr int MAX_TOP_LOGITS = 64;
constexpr int NATIVE_SCHEMA_VERSION = 1;
constexpr llama_state_seq_flags STATE_FLAGS = LLAMA_STATE_SEQ_FLAGS_KEEP_SWA_MASKED;

// Errors the caller can act on carry a protocol code. Anything else escaping
// an inference step is an execution_error and clears the contexts.
struct protocol_error : std::runtime_error {
    std::string code;
    std::string reason;
    protocol_error(std::string code_, std::string reason_, const std::string & message)
        : std::runtime_error(message), code(std::move(code_)), reason(std::move(reason_)) {}
};

protocol_error invalid(const std::string & reason, const std::string & message) {
    return protocol_error("invalid_request", reason, message);
}

struct settings {
    std::string model_path;
    std::string model_sha256;
    std::string runtime_sha256;
    std::string runtime_dir;
    int context = 2048;
    int batch = 256;
    int ubatch = 256;
    int threads = 8;
    bool gpu = false;
    bool reference = false;
    bool full = false;  // profile full-v1: one whole-model context

    const char * profile() const { return full ? FULL_PROFILE : PROFILE; }
    const char * execution_mode() const { return full ? "full" : "split18-30"; }
};

double elapsed_ms(const steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(steady_clock::now() - started).count();
}

int parse_positive(const std::string & text, const std::string & name) {
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    return value;
}

settings parse_args(int argc, char ** argv) {
    std::map<std::string, std::string> values;
    std::set<std::string> flags;
    const std::set<std::string> valued{
        "--model", "--model-sha256", "--runtime-sha256", "--runtime-dir",
        "--context", "--batch", "--ubatch", "--threads",
    };
    for (int index = 1; index < argc; ++index) {
        const std::string key(argv[index]);
        if (key == "--gpu" || key == "--reference") {
            if (!flags.insert(key).second) throw std::runtime_error("Repeated " + key);
            continue;
        }
        const bool optional = key == "--profile";
        if ((!valued.count(key) && !optional) || index + 1 >= argc || values.count(key)) {
            throw std::runtime_error("Invalid or repeated argument: " + key);
        }
        values[key] = argv[++index];
    }
    for (const auto & key : valued) {
        if (!values.count(key)) throw std::runtime_error("Missing " + key);
    }
    settings result;
    result.model_path = values.at("--model");
    result.model_sha256 = values.at("--model-sha256");
    result.runtime_sha256 = values.at("--runtime-sha256");
    result.runtime_dir = values.at("--runtime-dir");
    if (result.runtime_dir.empty()) throw std::runtime_error("runtime-dir must not be empty");
    result.context = parse_positive(values.at("--context"), "context");
    result.batch = parse_positive(values.at("--batch"), "batch");
    result.ubatch = parse_positive(values.at("--ubatch"), "ubatch");
    result.threads = parse_positive(values.at("--threads"), "threads");
    // One ubatch per decode keeps the per-call layer-input rows contiguous.
    if (result.ubatch != result.batch) throw std::runtime_error("ubatch must equal batch");
    result.gpu = flags.count("--gpu") > 0;
    result.reference = flags.count("--reference") > 0;
    const std::string profile = values.count("--profile") ? values.at("--profile") : PROFILE;
    if (profile != PROFILE && profile != FULL_PROFILE) throw std::runtime_error("Unknown profile " + profile);
    result.full = profile == FULL_PROFILE;
    // The full-v1 snapshot path already is the stock graph; the reference
    // context exists only to qualify the split against it.
    if (result.full && result.reference) throw std::runtime_error("--reference needs the split profile");
    return result;
}

void reject_incompatible_environment() {
    for (const auto * name : {
             "UNRIDDEN_GEMMA4_EXIT_LAYER",
             "GGML_CUDA_DISABLE_FUSION",
             "GGML_CUDA_DISABLE_GRAPHS"}) {
        if (std::getenv(name)) {
            throw std::runtime_error(std::string("Incompatible environment setting: ") + name);
        }
    }
}

// The helpers below mirror the v1 worker (native/worker.cpp) on purpose: the
// v1 source is hash-pinned by its manifest and stays untouched.
std::vector<llama_token> tokenize(
        const llama_vocab * vocab, const std::string & text, bool add_special = true) {
    const int count = llama_tokenize(
        vocab, text.data(), text.size(), nullptr, 0, add_special, true);
    if (count >= 0) throw std::runtime_error("Empty or invalid tokenization");
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    const int written = llama_tokenize(
        vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (written != static_cast<int>(tokens.size())) {
        throw std::runtime_error("Tokenization size changed");
    }
    return tokens;
}

void reject_control_tokens(const llama_vocab * vocab, const json & messages) {
    for (const auto & message : messages) {
        const std::string content = message.at("content").get<std::string>();
        if (content.empty()) continue;
        for (const llama_token token : tokenize(vocab, content, false)) {
            if (llama_vocab_is_control(vocab, token) || llama_vocab_is_eog(vocab, token)) {
                throw invalid("control_tokens", "Caller content contains a control token");
            }
        }
    }
}

void validate_messages(const json & messages) {
    if (!messages.is_array() || messages.empty() || messages.size() > 64) {
        throw invalid("internal", "messages must be a nonempty array of at most 64");
    }
    for (const auto & message : messages) {
        if (!message.is_object() || message.size() != 2) {
            throw invalid("internal", "a message has exactly role and content");
        }
        const std::string role = message.at("role").get<std::string>();
        if (role != "user" && role != "assistant") {
            throw invalid("internal", "message role must be user or assistant");
        }
        (void) message.at("content").get<std::string>();
    }
    if (messages.back().at("role") != "user") {
        throw invalid("internal", "the last message must be a user turn");
    }
}

std::string render_prompt(
        const json & messages,
        const common_chat_templates * templates,
        const std::string & answer_prefix) {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages.dump()));
    inputs.enable_thinking = false;
    inputs.chat_template_kwargs["enable_thinking"] = "false";
    inputs.now = std::chrono::system_clock::time_point{};
    return common_chat_templates_apply(templates, inputs).prompt + answer_prefix;
}

std::string sha256_hex(const unsigned char * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, data, size);
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) stream << std::setw(2) << static_cast<unsigned int>(byte);
    return stream.str();
}

std::string sha256_hex(const std::string & value) {
    return sha256_hex(reinterpret_cast<const unsigned char *>(value.data()), value.size());
}

bool read_bounded_line(std::istream & stream, std::string & line, size_t limit) {
    line.clear();
    char value = '\0';
    while (stream.get(value)) {
        if (value == '\n') return true;
        if (line.size() == limit) throw std::runtime_error("Protocol request exceeds size limit");
        line.push_back(value);
    }
    return !line.empty();
}

std::string base64(const unsigned char * data, size_t size) {
    static const char * table =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t index = 0; index < size; index += 3) {
        const uint32_t a = data[index];
        const uint32_t b = index + 1 < size ? data[index + 1] : 0;
        const uint32_t c = index + 2 < size ? data[index + 2] : 0;
        const uint32_t triple = (a << 16) | (b << 8) | c;
        out.push_back(table[(triple >> 18) & 63]);
        out.push_back(table[(triple >> 12) & 63]);
        out.push_back(index + 1 < size ? table[(triple >> 6) & 63] : '=');
        out.push_back(index + 2 < size ? table[triple & 63] : '=');
    }
    return out;
}

json vector_json(const float * rows, size_t row_count, size_t width, const std::string & representation) {
    for (size_t index = 0; index < row_count * width; ++index) {
        if (!std::isfinite(rows[index])) throw std::runtime_error("Nonfinite exported vector");
    }
    json shape = row_count == 1 ? json::array({width}) : json::array({row_count, width});
    return {
        {"dtype", "f32"},
        {"byte_order", "little"},
        {"shape", shape},
        {"representation", representation},
        {"base64", base64(reinterpret_cast<const unsigned char *>(rows), row_count * width * sizeof(float))},
    };
}

using blob = std::vector<uint8_t>;
using matrix = std::vector<float>;

struct readout {
    std::vector<double> label_logits;
    std::vector<int> label_token_ids;
    double allowed_mass = 0.0;
    int argmax_token = 0;
    double argmax_logit = 0.0;
    std::vector<std::pair<int, double>> top;
};

struct snapshot {
    std::string id;
    int completed_blocks = 0;
    std::string parent;
    std::string kind;  // "context" or "readout"
    std::vector<llama_token> tokens;
    std::string prompt_sha256;
    // Shared, never mutated after publication. A 30 snapshot made together
    // with (or promoted from) an 18 one holds the same lower blob pointer.
    std::shared_ptr<const blob> lower_kv;
    std::shared_ptr<const blob> upper_kv;
    std::shared_ptr<const matrix> h18;  // [N, n_embd], raw residual entering block 18
    std::shared_ptr<const matrix> h30;  // [N, n_embd], raw residual after block 29
    std::vector<float> last_normalized;  // [n_embd], head input at N-1
    bool lower_kv_owned = true;          // counts the lower blob in this row's bytes
};

json snapshot_row(const snapshot & item) {
    const auto size_of = [](const auto & pointer) -> size_t {
        return pointer ? pointer->size() : 0;
    };
    return {
        {"snapshot_id", item.id},
        {"completed_blocks", item.completed_blocks},
        {"parent", item.parent.empty() ? json() : json(item.parent)},
        {"kind", item.kind},
        {"tokens", item.tokens.size()},
        {"prompt_sha256", item.prompt_sha256},
        {"bytes", {
            {"lower_kv", item.lower_kv_owned ? size_of(item.lower_kv) : 0},
            {"upper_kv", size_of(item.upper_kv)},
            {"h18", size_of(item.h18) * sizeof(float)},
            {"h30", size_of(item.h30) * sizeof(float)},
        }},
    };
}

struct batch_holder {
    llama_batch batch;
    batch_holder(int32_t capacity, int32_t embd) : batch(llama_batch_init(capacity, embd, 1)) {}
    ~batch_holder() { llama_batch_free(batch); }
    batch_holder(const batch_holder &) = delete;
    batch_holder & operator=(const batch_holder &) = delete;
};

using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

class runtime {
public:
    runtime(const settings & config, llama_model * model)
        : config_(config), model_(model), vocab_(llama_model_get_vocab(model)),
          n_embd_(llama_model_n_embd(model)), n_layer_(llama_model_n_layer(model)),
          lower_(nullptr, llama_free), upper_(nullptr, llama_free), reference_(nullptr, llama_free) {
        if (config.full) {
            // Default range: the stock graph, so shared K/V and per-layer
            // inputs never cross a boundary. Same fixed-capture rule.
            lower_ = make_context(0, -1);
            llama_set_embeddings_layer_inp(lower_.get(), static_cast<uint32_t>(n_layer_), true);
            llama_set_embeddings_nextn(lower_.get(), true, false);
            return;
        }
        lower_ = make_context(0, SPLIT_BLOCK);
        upper_ = make_context(SPLIT_BLOCK, n_layer_);
        // Captures are fixed for each context's lifetime, so create, promote
        // and every branch run the very same graphs.
        llama_set_embeddings_layer_inp(lower_.get(), SPLIT_BLOCK, true);
        llama_set_embeddings_layer_inp(upper_.get(), static_cast<uint32_t>(n_layer_), true);
        llama_set_embeddings_nextn(upper_.get(), true, false);
        if (config.reference) {
            reference_ = make_context(0, -1);
            llama_set_embeddings_layer_inp(reference_.get(), SPLIT_BLOCK, true);
            llama_set_embeddings_layer_inp(reference_.get(), static_cast<uint32_t>(n_layer_), true);
        }
    }

    int n_embd() const { return n_embd_; }
    int n_layer() const { return n_layer_; }
    bool full() const { return config_.full; }
    const llama_vocab * vocab() const { return vocab_; }
    bool has_reference() const { return reference_ != nullptr; }

    // full-v1: every block over `tokens[begin:end)`, appended to seq 0.
    // Returns the final-block residual rows; fills the last head input and,
    // when asked, the last position's vocabulary logits.
    matrix run_full(const std::vector<llama_token> & tokens, size_t begin, size_t end,
                    std::vector<float> & last_normalized, std::vector<double> * logits) {
        if (end <= begin || end > tokens.size()) throw std::runtime_error("Bad token range");
        auto * context = lower_.get();
        const size_t width = static_cast<size_t>(n_embd_);
        matrix rows;
        rows.reserve((end - begin) * width);
        size_t offset = begin;
        for (const size_t count : unridden::prefill_chunks(end - begin, static_cast<size_t>(config_.batch))) {
            auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + offset,
                                             static_cast<int32_t>(count));
            if (llama_decode(context, batch) != 0) throw std::runtime_error("Full-depth decode failed");
            llama_synchronize(context);
            append_layer_rows(context, static_cast<uint32_t>(n_layer_), count, rows);
            offset += count;
            if (offset == end) {
                const float * head = llama_get_embeddings_nextn_ith(context, static_cast<int32_t>(count - 1));
                if (!head) throw std::runtime_error("Missing head input");
                last_normalized.assign(head, head + width);
                if (logits) {
                    const float * raw = llama_get_logits_ith(context, -1);
                    if (!raw) throw std::runtime_error("Missing full-depth logits");
                    logits->assign(raw, raw + llama_vocab_n_tokens(vocab_));
                }
            }
        }
        return rows;
    }

    // Lower blocks over `tokens[begin:end)`, appended to lower seq 0. Returns
    // the H18 rows of exactly those tokens.
    matrix run_lower(const std::vector<llama_token> & tokens, size_t begin, size_t end) {
        return run_tokens(lower_.get(), tokens, begin, end, SPLIT_BLOCK, nullptr);
    }

    // Upper blocks over boundary rows for positions [first_pos, first_pos+rows).
    // Returns H30 rows; fills the last-position head input and, when asked,
    // copies the last position's vocabulary logits.
    matrix run_upper(const matrix & h18_rows, llama_pos first_pos,
                     std::vector<float> & last_normalized, std::vector<double> * logits) {
        const size_t width = static_cast<size_t>(n_embd_);
        const size_t rows = h18_rows.size() / width;
        if (rows == 0 || rows * width != h18_rows.size()) throw std::runtime_error("Bad boundary rows");
        auto * context = upper_.get();
        matrix h30;
        h30.reserve(h18_rows.size());
        batch_holder holder(config_.batch, n_embd_);
        size_t offset = 0;
        for (const size_t count : unridden::prefill_chunks(rows, static_cast<size_t>(config_.batch))) {
            llama_batch & batch = holder.batch;
            batch.n_tokens = static_cast<int32_t>(count);
            std::memcpy(batch.embd, h18_rows.data() + offset * width, count * width * sizeof(float));
            const bool final_chunk = offset + count == rows;
            for (size_t slot = 0; slot < count; ++slot) {
                batch.pos[slot] = first_pos + static_cast<llama_pos>(offset + slot);
                batch.n_seq_id[slot] = 1;
                batch.seq_id[slot][0] = 0;
                batch.logits[slot] = final_chunk && slot + 1 == count ? 1 : 0;
            }
            if (llama_decode(context, batch) != 0) throw std::runtime_error("Upper-range decode failed");
            llama_synchronize(context);
            append_layer_rows(context, static_cast<uint32_t>(n_layer_), count, h30);
            if (final_chunk) {
                const float * head = llama_get_embeddings_nextn_ith(context, static_cast<int32_t>(count - 1));
                if (!head) throw std::runtime_error("Missing head input");
                last_normalized.assign(head, head + width);
                if (logits) {
                    const float * raw = llama_get_logits_ith(context, -1);
                    if (!raw) throw std::runtime_error("Missing upper-range logits");
                    const int vocabulary = llama_vocab_n_tokens(vocab_);
                    logits->assign(raw, raw + vocabulary);
                }
            }
            offset += count;
        }
        return h30;
    }

    // Stock uninterrupted 30 blocks from an empty cache, same chunking.
    void run_reference(const std::vector<llama_token> & tokens, matrix & h18, matrix & h30,
                       std::vector<double> & logits) {
        if (!reference_) throw protocol_error("capability_unavailable", "", "no reference context");
        auto * context = reference_.get();
        llama_memory_clear(llama_get_memory(context), true);
        h18.clear();
        h30.clear();
        size_t offset = 0;
        for (const size_t count : unridden::prefill_chunks(tokens.size(), static_cast<size_t>(config_.batch))) {
            auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + offset,
                                             static_cast<int32_t>(count));
            if (llama_decode(context, batch) != 0) throw std::runtime_error("Reference decode failed");
            llama_synchronize(context);
            append_layer_rows(context, SPLIT_BLOCK, count, h18);
            append_layer_rows(context, static_cast<uint32_t>(n_layer_), count, h30);
            offset += count;
        }
        const float * raw = llama_get_logits_ith(context, -1);
        if (!raw) throw std::runtime_error("Missing reference logits");
        logits.assign(raw, raw + llama_vocab_n_tokens(vocab_));
        llama_memory_clear(llama_get_memory(context), true);
    }

    // Stock 30 blocks over tokens[begin:end), appended to the reference cache
    // (cleared first when asked). Copies the last position's logits.
    void reference_append(const std::vector<llama_token> & tokens, size_t begin, size_t end, bool fresh,
                          std::vector<double> & logits) {
        if (!reference_) throw protocol_error("capability_unavailable", "", "no reference context");
        auto * context = reference_.get();
        if (fresh) llama_memory_clear(llama_get_memory(context), true);
        size_t offset = begin;
        for (const size_t count : unridden::prefill_chunks(end - begin, static_cast<size_t>(config_.batch))) {
            auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + offset,
                                             static_cast<int32_t>(count));
            if (llama_decode(context, batch) != 0) throw std::runtime_error("Reference decode failed");
            offset += count;
        }
        llama_synchronize(context);
        const float * raw = llama_get_logits_ith(context, -1);
        if (!raw) throw std::runtime_error("Missing reference logits");
        logits.assign(raw, raw + llama_vocab_n_tokens(vocab_));
    }

    void clear_reference() {
        if (reference_) llama_memory_clear(llama_get_memory(reference_.get()), true);
    }

    blob save_state(bool upper) {
        auto * context = upper ? upper_.get() : lower_.get();
        // Every cell, SWA-masked ones included: the restored cache then has
        // the same cells in the same slots as the live one, and the same math.
        const size_t size = llama_state_seq_get_size_ext(context, 0, STATE_FLAGS);
        blob data(size);
        if (llama_state_seq_get_data_ext(context, data.data(), size, 0, STATE_FLAGS) != size) {
            throw std::runtime_error("Sequence state size changed during save");
        }
        return data;
    }

    void clear() {
        llama_memory_clear(llama_get_memory(lower_.get()), true);
        if (upper_) llama_memory_clear(llama_get_memory(upper_.get()), true);
        resident_.clear();
        resident_upper_ = false;
    }

    // Leaves seq 0 of lower (and upper, when the snapshot has it) holding the
    // snapshot's exact tokens. Returns restored bytes; 0 means it was resident.
    size_t make_resident(const snapshot & item) {
        const auto held = static_cast<llama_pos>(item.tokens.size()) - 1;
        const bool want_upper = !config_.full && item.completed_blocks == n_layer_;
        if (resident_ == item.id && (!want_upper || resident_upper_) &&
                llama_memory_seq_pos_max(llama_get_memory(lower_.get()), 0) == held &&
                (!want_upper || llama_memory_seq_pos_max(llama_get_memory(upper_.get()), 0) == held)) {
            return 0;
        }
        clear();
        size_t restored = restore(lower_.get(), *item.lower_kv);
        if (want_upper) restored += restore(upper_.get(), *item.upper_kv);
        if (llama_memory_seq_pos_max(llama_get_memory(lower_.get()), 0) != held ||
                (want_upper && llama_memory_seq_pos_max(llama_get_memory(upper_.get()), 0) != held)) {
            clear();
            throw std::runtime_error("Restored state does not hold the snapshot tokens");
        }
        resident_ = item.id;
        resident_upper_ = want_upper;
        return restored;
    }

    // Drops everything past the resident snapshot's tokens.
    void trim_to(const snapshot & item) {
        const auto length = static_cast<llama_pos>(item.tokens.size());
        for (auto * context : {lower_.get(), upper_.get()}) {
            if (!context) continue;
            auto * memory = llama_get_memory(context);
            if (llama_memory_seq_pos_max(memory, 0) >= length &&
                    !llama_memory_seq_rm(memory, 0, length, -1)) {
                clear();
                throw std::runtime_error("Context memory could not be trimmed to the parent");
            }
        }
        if (llama_memory_seq_pos_max(llama_get_memory(lower_.get()), 0) != length - 1) {
            clear();
            throw std::runtime_error("Trim left unexpected lower state");
        }
    }

    void set_resident(const std::string & id, bool upper) {
        resident_ = id;
        resident_upper_ = upper;
    }

    const std::string & resident() const { return resident_; }

private:
    context_ptr make_context(int32_t begin, int32_t end) {
        auto params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(config_.context);
        params.n_batch = static_cast<uint32_t>(config_.batch);
        params.n_ubatch = static_cast<uint32_t>(config_.ubatch);
        params.n_seq_max = 1;
        params.kv_unified = false;
        params.n_threads = config_.threads;
        params.n_threads_batch = config_.threads;
        params.swa_full = true;
        params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        params.offload_kqv = config_.gpu;
        params.op_offload = config_.gpu;
        params.no_perf = false;
        params.layer_begin = begin;
        params.layer_end = end;
        context_ptr context(llama_init_from_model(model_, params), llama_free);
        if (!context) throw std::runtime_error("Context initialization failed");
        return context;
    }

    matrix run_tokens(llama_context * context, const std::vector<llama_token> & tokens,
                      size_t begin, size_t end, uint32_t capture, std::vector<double> *) {
        if (end <= begin || end > tokens.size()) throw std::runtime_error("Bad token range");
        matrix rows;
        rows.reserve((end - begin) * static_cast<size_t>(n_embd_));
        size_t offset = begin;
        for (const size_t count : unridden::prefill_chunks(end - begin, static_cast<size_t>(config_.batch))) {
            auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + offset,
                                             static_cast<int32_t>(count));
            if (llama_decode(context, batch) != 0) throw std::runtime_error("Lower-range decode failed");
            llama_synchronize(context);
            append_layer_rows(context, capture, count, rows);
            offset += count;
        }
        return rows;
    }

    void append_layer_rows(llama_context * context, uint32_t layer, size_t count, matrix & out) {
        const float * data = llama_get_embeddings_layer_inp(context, layer);
        if (!data) throw std::runtime_error("Missing layer capture");
        const size_t floats = count * static_cast<size_t>(n_embd_);
        for (size_t index = 0; index < floats; ++index) {
            if (!std::isfinite(data[index])) throw std::runtime_error("Nonfinite residual capture");
        }
        out.insert(out.end(), data, data + floats);
    }

    size_t restore(llama_context * context, const blob & data) {
        if (llama_state_seq_set_data_ext(context, data.data(), data.size(), 0, STATE_FLAGS) == 0) {
            clear();
            throw std::runtime_error("Sequence state restore failed");
        }
        return data.size();
    }

    const settings & config_;
    llama_model * model_;
    const llama_vocab * vocab_;
    int n_embd_;
    int n_layer_;
    context_ptr lower_;
    context_ptr upper_;
    context_ptr reference_;
    std::string resident_;
    bool resident_upper_ = false;
};

readout make_readout(const std::vector<double> & vocabulary, const std::vector<int> & label_ids, int top_k) {
    readout result;
    if (!label_ids.empty()) {
        for (const int token_id : label_ids) {
            result.label_logits.push_back(vocabulary.at(static_cast<size_t>(token_id)));
        }
        (void) unridden::stable_softmax(result.label_logits);
        const auto summary = unridden::summarize_vocabulary(vocabulary, label_ids);
        result.label_token_ids = label_ids;
        result.allowed_mass = summary.allowed_mass;
        result.argmax_token = summary.argmax_token_id;
        result.argmax_logit = summary.argmax_logit;
    } else {
        const auto argmax = std::max_element(vocabulary.begin(), vocabulary.end());
        result.argmax_token = static_cast<int>(std::distance(vocabulary.begin(), argmax));
        result.argmax_logit = *argmax;
    }
    if (top_k > 0) {
        std::vector<int> order(vocabulary.size());
        for (size_t index = 0; index < order.size(); ++index) order[index] = static_cast<int>(index);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
            return vocabulary[static_cast<size_t>(a)] > vocabulary[static_cast<size_t>(b)];
        });
        for (int index = 0; index < top_k; ++index) {
            result.top.emplace_back(order[static_cast<size_t>(index)],
                                    vocabulary[static_cast<size_t>(order[static_cast<size_t>(index)])]);
        }
    }
    return result;
}

json top_json(const readout & value) {
    json rows = json::array();
    for (const auto & [token_id, logit] : value.top) rows.push_back({{"token_id", token_id}, {"logit", logit}});
    return rows;
}

json readout_json(const readout & value) {
    return {
        {"label_logits", value.label_logits},
        {"label_token_ids", value.label_token_ids},
        {"allowed_label_mass", value.allowed_mass},
        {"full_vocabulary_argmax", {{"token_id", value.argmax_token}, {"logit", value.argmax_logit}}},
        {"top_logits", top_json(value)},
    };
}

struct rendered {
    std::string prompt;
    std::vector<llama_token> tokens;
};

class worker {
public:
    worker(const settings & config, llama_model * model, const common_chat_templates * templates,
           std::vector<std::string> labels, std::vector<int> label_ids)
        : config_(config), runtime_(config, model), templates_(templates),
          labels_(std::move(labels)), label_ids_(std::move(label_ids)) {}

    json handle(const json & request) {
        const std::string type = request.at("type").get<std::string>();
        if (type == "create") return create(request);
        if (type == "promote") return promote(request);
        if (type == "evaluate") return evaluate(request);
        if (type == "state_eval") return state_eval(request);
        if (type == "inspect") return inspect(request);
        if (type == "vectors") return vectors(request);
        if (type == "drop") return drop(request);
        if (type == "save") return save(request);
        if (type == "load") return load(request);
        if (type == "reference") return reference(request);
        if (type == "ride") return ride(request);
        throw invalid("internal", "Unknown request type");
    }

    void reset_contexts() { runtime_.clear(); }

private:
    // -- rendering -------------------------------------------------------

    rendered render(const json & messages, const std::string & answer_prefix) {
        validate_messages(messages);
        reject_control_tokens(runtime_.vocab(), messages);
        rendered result;
        result.prompt = render_prompt(messages, templates_, answer_prefix);
        result.tokens = tokenize(runtime_.vocab(), result.prompt);
        const llama_token bos = llama_vocab_bos(runtime_.vocab());
        if (result.tokens.size() > 1 && result.tokens[0] == bos && result.tokens[1] == bos) {
            throw invalid("internal", "Rendered prompt starts with a doubled BOS");
        }
        if (result.tokens.size() > static_cast<size_t>(config_.context)) {
            throw invalid("budget", "Prompt exceeds context size");
        }
        return result;
    }

    // Frozen tokens for a context freeze: the prompt through the marked bytes
    // of the last user message, minus the final token, which could merge with
    // whatever follows. Same rule as the v1 shared-prefix split.
    std::vector<llama_token> context_prefix(const json & messages, const rendered & full, size_t bytes,
                                            std::string & frozen_text) {
        std::string content;
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (it->at("role") == "user") {
                content = it->at("content").get<std::string>();
                break;
            }
        }
        if (bytes == 0 || bytes > content.size()) throw invalid("internal", "content_bytes out of range");
        // Templates may trim a turn's trailing whitespace, so the marker is
        // matched without it; branches still render it, after the freeze point.
        std::string marker = content.substr(0, bytes);
        while (!marker.empty() && std::isspace(static_cast<unsigned char>(marker.back()))) marker.pop_back();
        if (marker.empty()) throw invalid("internal", "content_bytes marks only whitespace");
        const size_t start = full.prompt.find(marker);
        if (start == std::string::npos) throw invalid("internal", "Marked content not found in prompt");
        frozen_text = full.prompt.substr(0, start + marker.size());
        const auto head = tokenize(runtime_.vocab(), frozen_text);
        const size_t limit = std::min(head.size(), full.tokens.size()) - 1;
        size_t common = 0;
        while (common < limit && head.at(common) == full.tokens.at(common)) ++common;
        if (common < 1) throw invalid("internal", "Context freeze point is empty");
        return {full.tokens.begin(), full.tokens.begin() + static_cast<std::ptrdiff_t>(common)};
    }

    std::vector<int> label_ids_for(const json & labels_value, const rendered & full) {
        if (labels_value.is_null()) return {};
        const auto labels = labels_value.get<std::vector<std::string>>();
        if (labels.size() < 2 || labels.size() > labels_.size() ||
                !std::equal(labels.begin(), labels.end(), labels_.begin())) {
            throw invalid("internal", "Labels differ from validated alphabet prefix");
        }
        std::vector<int> ids;
        for (size_t index = 0; index < labels.size(); ++index) {
            const auto extended = tokenize(runtime_.vocab(), full.prompt + labels.at(index));
            const int token_id = unridden::single_suffix_token(full.tokens, extended);
            if (token_id != label_ids_.at(index)) {
                throw invalid("internal", "Prompt-specific label token mapping changed");
            }
            ids.push_back(token_id);
        }
        return ids;
    }

    int top_k(const json & request) {
        const int value = request.value("top_logits", 0);
        if (value < 0 || value > MAX_TOP_LOGITS) throw invalid("internal", "top_logits out of range");
        return value;
    }

    // The tag on the `h30` slot: in the split profile block 30 is the last.
    const char * final_representation() const {
        return runtime_.full() ? "raw_residual_after_final_block" : "raw_residual_after_block_30";
    }

    // -- registry --------------------------------------------------------

    const snapshot & find(const std::string & id) {
        const auto it = snapshots_.find(id);
        if (it == snapshots_.end()) throw protocol_error("snapshot_not_found", "", "Unknown snapshot " + id);
        return it->second;
    }

    void require_fresh(const std::string & id) {
        if (id.empty() || id.size() > 128) throw invalid("internal", "Invalid snapshot id");
        for (const char c : id) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
                throw invalid("internal", "Invalid snapshot id");
            }
        }
        if (snapshots_.count(id)) throw protocol_error("snapshot_exists", "", "Snapshot id exists " + id);
    }

    // Branch suffix of `full` after the parent's frozen tokens.
    size_t branch_suffix(const snapshot & parent, const rendered & full) {
        const size_t n = parent.tokens.size();
        if (full.tokens.size() <= n ||
                !std::equal(parent.tokens.begin(), parent.tokens.end(), full.tokens.begin())) {
            if (parent.kind == "readout") {
                throw protocol_error("followup_unsupported", "",
                                     "The template cannot continue this readout snapshot");
            }
            throw protocol_error("snapshot_prefix_mismatch", "", "Branch does not extend the snapshot");
        }
        return full.tokens.size() - n;
    }

    // -- commands --------------------------------------------------------

    // The frozen tokens and prompt hash of a create request.
    struct frozen {
        rendered full;
        std::vector<llama_token> tokens;
        std::string text;
        std::vector<int> label_ids;
    };

    frozen freeze_tokens(const json & request, const std::string & kind) {
        frozen result;
        result.full = render(request.at("messages"), request.at("answer_prefix").get<std::string>());
        if (kind == "context") {
            if (!request.at("labels").is_null()) throw invalid("internal", "labels need a readout freeze");
            result.tokens = context_prefix(request.at("messages"), result.full,
                                           request.at("freeze").at("content_bytes").get<size_t>(), result.text);
        } else if (kind == "readout") {
            result.tokens = result.full.tokens;
            result.text = result.full.prompt;
            result.label_ids = label_ids_for(request.at("labels"), result.full);
        } else {
            throw invalid("internal", "Unknown freeze kind");
        }
        return result;
    }

    // full-v1: one pass over every block, one snapshot at n_layer.
    json create_full(const json & request) {
        const std::string kind = request.at("freeze").at("kind").get<std::string>();
        const auto & checkpoints = request.at("checkpoints");
        const std::string final_key = std::to_string(runtime_.n_layer());
        if (!checkpoints.is_object() || checkpoints.size() != 1 || !checkpoints.contains(final_key)) {
            throw invalid("internal", "checkpoints must be exactly " + final_key);
        }
        const std::string id = checkpoints.at(final_key).get<std::string>();
        require_fresh(id);
        const frozen input = freeze_tokens(request, kind);
        const int k = top_k(request);
        const auto started = steady_clock::now();
        json response;
        try {
            runtime_.clear();
            snapshot item;
            std::vector<double> vocabulary;
            item.h30 = std::make_shared<const matrix>(
                runtime_.run_full(input.tokens, 0, input.tokens.size(), item.last_normalized, &vocabulary));
            const double run_ms = elapsed_ms(started);
            item.id = id;
            item.completed_blocks = runtime_.n_layer();
            item.kind = kind;
            item.tokens = input.tokens;
            item.prompt_sha256 = sha256_hex(input.text);
            item.lower_kv = std::make_shared<const blob>(runtime_.save_state(false));
            const readout value = make_readout(vocabulary, input.label_ids, k);
            const json readout_value = (input.label_ids.empty() && k == 0) ? json() : readout_json(value);
            const json row = snapshot_row(item);
            snapshots_.emplace(id, std::move(item));
            runtime_.set_resident(id, false);
            response = {
                {"type", "created"},
                {"prompt_sha256", row.at("prompt_sha256")},
                {"tokens", input.tokens.size()},
                {"snapshots", json::array({row})},
                {"readout", readout_value},
                {"block_tokens", {{"lower", input.tokens.size()}, {"upper", 0}}},
                {"timing_ms", {{"lower", run_ms}, {"upper", 0.0}, {"total", elapsed_ms(started)}}},
                {"generated_tokens", 0},
            };
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            throw protocol_error("execution_error", "", error.what());
        }
        return response;
    }

    json create(const json & request) {
        if (runtime_.full()) return create_full(request);
        const auto & freeze = request.at("freeze");
        const std::string kind = freeze.at("kind").get<std::string>();
        const auto & checkpoints = request.at("checkpoints");
        std::string id18;
        std::string id30;
        for (const auto & [key, value] : checkpoints.items()) {
            if (key == "18") id18 = value.get<std::string>();
            else if (key == "30") id30 = value.get<std::string>();
            else throw invalid("internal", "checkpoints must be 18 and/or 30");
        }
        if (id18.empty() && id30.empty()) throw invalid("internal", "No checkpoint requested");
        if (!id18.empty()) require_fresh(id18);
        if (!id30.empty()) require_fresh(id30);
        if (id18 == id30) throw invalid("internal", "Checkpoint ids must differ");
        const std::string answer_prefix = request.at("answer_prefix").get<std::string>();
        const rendered full = render(request.at("messages"), answer_prefix);
        std::vector<llama_token> tokens;
        std::string frozen_text;
        std::vector<int> label_ids;
        if (kind == "context") {
            if (!request.at("labels").is_null()) throw invalid("internal", "labels need a readout freeze");
            tokens = context_prefix(request.at("messages"), full, freeze.at("content_bytes").get<size_t>(),
                                    frozen_text);
        } else if (kind == "readout") {
            tokens = full.tokens;
            frozen_text = full.prompt;
            label_ids = label_ids_for(request.at("labels"), full);
            if (!label_ids.empty() && id30.empty()) {
                throw invalid("internal", "labels need a 30 checkpoint");
            }
        } else {
            throw invalid("internal", "Unknown freeze kind");
        }
        const int k = top_k(request);
        if (k > 0 && id30.empty()) throw invalid("internal", "top_logits need a 30 checkpoint");

        const auto started = steady_clock::now();
        json response;
        try {
            runtime_.clear();
            auto mark = steady_clock::now();
            auto h18 = std::make_shared<const matrix>(runtime_.run_lower(tokens, 0, tokens.size()));
            const double lower_ms = elapsed_ms(mark);
            auto lower_kv = std::make_shared<const blob>(runtime_.save_state(false));
            snapshot s18;
            s18.id = id18;
            s18.completed_blocks = 18;
            s18.kind = kind;
            s18.tokens = tokens;
            s18.prompt_sha256 = sha256_hex(frozen_text);
            s18.lower_kv = lower_kv;
            s18.h18 = h18;
            json rows = json::array();
            json readout_value;
            double upper_ms = 0.0;
            size_t upper_tokens = 0;
            snapshot s30;
            if (!id30.empty()) {
                mark = steady_clock::now();
                std::vector<double> vocabulary;
                s30.h30 = std::make_shared<const matrix>(
                    runtime_.run_upper(*h18, 0, s30.last_normalized, &vocabulary));
                upper_ms = elapsed_ms(mark);
                upper_tokens = tokens.size();
                s30.id = id30;
                s30.completed_blocks = 30;
                s30.kind = kind;
                s30.tokens = tokens;
                s30.prompt_sha256 = s18.prompt_sha256;
                s30.lower_kv = lower_kv;
                s30.upper_kv = std::make_shared<const blob>(runtime_.save_state(true));
                if (!id18.empty()) {
                    s30.parent = id18;
                    s30.lower_kv_owned = false;
                } else {
                    // Without a published 18 the 30 row keeps H18 itself, so
                    // a later branch or inspection still has the boundary.
                    s30.h18 = h18;
                }
                const readout value = make_readout(vocabulary, label_ids, k);
                readout_value = (label_ids.empty() && k == 0) ? json() : readout_json(value);
            }
            // Publish only after every requested checkpoint is complete.
            if (!id18.empty()) {
                rows.push_back(snapshot_row(s18));
                snapshots_.emplace(id18, s18);
            }
            if (!id30.empty()) {
                rows.push_back(snapshot_row(s30));
                snapshots_.emplace(id30, s30);
                runtime_.set_resident(id30, true);
            } else {
                runtime_.set_resident(id18, false);
            }
            response = {
                {"type", "created"},
                {"prompt_sha256", s18.prompt_sha256},
                {"tokens", tokens.size()},
                {"snapshots", rows},
                {"readout", readout_value},
                {"block_tokens", {{"lower", tokens.size()}, {"upper", upper_tokens}}},
                {"timing_ms", {{"lower", lower_ms}, {"upper", upper_ms}, {"total", elapsed_ms(started)}}},
                {"generated_tokens", 0},
            };
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            throw protocol_error("execution_error", "", error.what());
        }
        return response;
    }

    json promote(const json & request) {
        const snapshot & parent = find(request.at("snapshot_id").get<std::string>());
        if (runtime_.full() || parent.completed_blocks != SPLIT_BLOCK || !parent.h18) {
            throw protocol_error("capability_unavailable", "", "Only an 18 snapshot can be promoted");
        }
        const std::string new_id = request.at("new_id").get<std::string>();
        require_fresh(new_id);
        const auto started = steady_clock::now();
        snapshot child;
        try {
            // Upper blocks only: the lower K/V is the parent's, by reference.
            runtime_.clear();
            runtime_.make_resident(parent);
            child.h30 = std::make_shared<const matrix>(
                runtime_.run_upper(*parent.h18, 0, child.last_normalized, nullptr));
            child.upper_kv = std::make_shared<const blob>(runtime_.save_state(true));
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            throw protocol_error("execution_error", "", error.what());
        }
        child.id = new_id;
        child.completed_blocks = 30;
        child.parent = parent.id;
        child.kind = parent.kind;
        child.tokens = parent.tokens;
        child.prompt_sha256 = parent.prompt_sha256;
        child.lower_kv = parent.lower_kv;
        child.lower_kv_owned = false;
        snapshots_.emplace(new_id, child);
        runtime_.set_resident(new_id, true);
        return {
            {"type", "promoted"},
            {"snapshot", snapshot_row(child)},
            {"block_tokens", {{"lower", 0}, {"upper", child.tokens.size()}}},
            {"timing_ms", {{"upper", elapsed_ms(started)}, {"total", elapsed_ms(started)}}},
            {"generated_tokens", 0},
        };
    }

    // One branch: parent resident, suffix through both ranges, readout, then
    // either publish a child or trim back. Returns the per-branch accounting.
    struct branch_result {
        std::vector<double> vocabulary;
        std::vector<float> last_normalized;
        std::vector<float> last_residual;
        json accounting;
        json child;
    };

    branch_result run_branch(const snapshot & parent, const rendered & full, size_t suffix,
                             const std::string & save_as) {
        branch_result result;
        const size_t n = parent.tokens.size();
        auto mark = steady_clock::now();
        const std::string previous_resident = runtime_.resident();
        const size_t restored = runtime_.make_resident(parent);
        const double restore_ms = elapsed_ms(mark);
        mark = steady_clock::now();
        const bool whole = runtime_.full();
        const matrix h18_q = whole ? matrix() : runtime_.run_lower(full.tokens, n, full.tokens.size());
        const matrix h30_q = whole
            ? runtime_.run_full(full.tokens, n, full.tokens.size(), result.last_normalized, &result.vocabulary)
            : runtime_.run_upper(h18_q, static_cast<llama_pos>(n), result.last_normalized, &result.vocabulary);
        const double inference_ms = elapsed_ms(mark);
        const size_t width = static_cast<size_t>(runtime_.n_embd());
        result.last_residual.assign(h30_q.end() - static_cast<std::ptrdiff_t>(width), h30_q.end());
        if (!save_as.empty()) {
            snapshot child;
            child.id = save_as;
            child.completed_blocks = runtime_.n_layer();
            child.parent = parent.id;
            child.kind = "readout";
            child.tokens = full.tokens;
            child.prompt_sha256 = sha256_hex(full.prompt);
            child.lower_kv = std::make_shared<const blob>(runtime_.save_state(false));
            if (!whole) child.upper_kv = std::make_shared<const blob>(runtime_.save_state(true));
            if (parent.h18) {
                matrix joined(*parent.h18);
                joined.insert(joined.end(), h18_q.begin(), h18_q.end());
                child.h18 = std::make_shared<const matrix>(std::move(joined));
            }
            if (parent.h30) {
                matrix joined(*parent.h30);
                joined.insert(joined.end(), h30_q.begin(), h30_q.end());
                child.h30 = std::make_shared<const matrix>(std::move(joined));
            }
            child.last_normalized = result.last_normalized;
            result.child = snapshot_row(child);
            snapshots_.emplace(save_as, std::move(child));
        }
        runtime_.trim_to(parent);
        result.accounting = {
            {"parent", parent.id},
            {"suffix_tokens", suffix},
            {"block_tokens", {{"lower", suffix}, {"upper", whole ? 0 : suffix}}},
            {"restore", restored == 0 ? "resident" : "host"},
            {"restored_bytes", restored},
            {"restore_ms", restore_ms},
            {"inference_ms", inference_ms},
            {"child", result.child.is_null() ? json() : result.child},
        };
        (void) previous_resident;
        return result;
    }

    json evaluate(const json & request) {
        const snapshot & parent = find(request.at("snapshot_id").get<std::string>());
        if (request.at("readout_blocks").get<int>() != runtime_.n_layer()) {
            throw protocol_error("capability_unavailable", "", "No registered early head for that block");
        }
        if (parent.completed_blocks != runtime_.n_layer()) {
            throw protocol_error("capability_unavailable", "", "Evaluate needs a 30 snapshot; promote first");
        }
        const auto & rows = request.at("questions");
        if (!rows.is_array() || rows.empty() || rows.size() > 32) throw invalid("internal", "Invalid question count");
        struct prepared {
            std::string id;
            rendered full;
            size_t suffix;
            std::vector<int> label_ids;
            std::string save_as;
        };
        std::vector<prepared> questions;
        std::set<std::string> ids;
        std::set<std::string> new_ids;
        // Preflight every question before any inference.
        for (const auto & row : rows) {
            prepared item;
            item.id = row.at("id").get<std::string>();
            if (item.id.empty() || !ids.insert(item.id).second) throw invalid("internal", "Bad question id");
            if (row.at("prompt_version").get<std::string>() != CONTEXT_PROMPT_VERSION) {
                throw invalid("internal", "Unsupported prompt version");
            }
            item.full = render(row.at("messages"), row.at("answer_prefix").get<std::string>());
            item.suffix = branch_suffix(parent, item.full);
            item.label_ids = label_ids_for(row.at("labels"), item.full);
            if (item.label_ids.empty()) throw invalid("internal", "A question needs labels");
            if (!row.at("save_as").is_null()) {
                item.save_as = row.at("save_as").get<std::string>();
                require_fresh(item.save_as);
                if (!new_ids.insert(item.save_as).second) throw invalid("internal", "Duplicate save_as");
            }
            questions.push_back(std::move(item));
        }
        json results = json::array();
        try {
            for (const auto & item : questions) {
                const auto started = steady_clock::now();
                auto branch = run_branch(parent, item.full, item.suffix, item.save_as);
                const readout value = make_readout(branch.vocabulary, item.label_ids, 0);
                results.push_back({
                    {"id", item.id},
                    {"label_logits", value.label_logits},
                    {"label_token_ids", value.label_token_ids},
                    {"allowed_label_mass", value.allowed_mass},
                    {"full_vocabulary_argmax", {{"token_id", value.argmax_token}, {"logit", value.argmax_logit}}},
                    {"prompt_sha256", sha256_hex(item.full.prompt)},
                    {"prompt_tokens", item.full.tokens.size()},
                    {"processed_tokens", item.suffix},
                    {"reused_tokens", parent.tokens.size()},
                    {"cache_cleared", false},
                    {"evaluation_mode", "sequential"},
                    {"batch_sequences", 1},
                    {"timing_ms", elapsed_ms(started)},
                    {"snapshot", branch.accounting},
                });
            }
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            throw protocol_error("execution_error", "", error.what());
        }
        return {
            {"type", "result"},
            {"model_sha256", config_.model_sha256},
            {"runtime_sha256", config_.runtime_sha256},
            {"generated_tokens", 0},
            {"callbacks_enabled", false},
            {"execution_mode", config_.execution_mode()},
            {"questions", results},
        };
    }

    json state_eval(const json & request) {
        const snapshot & parent = find(request.at("snapshot_id").get<std::string>());
        if (parent.completed_blocks != runtime_.n_layer()) {
            throw protocol_error("capability_unavailable", "", "state_eval needs a 30 snapshot; promote first");
        }
        const rendered full = render(request.at("messages"), request.at("answer_prefix").get<std::string>());
        const size_t suffix = branch_suffix(parent, full);
        std::set<std::string> exports;
        for (const auto & item : request.at("export")) {
            const std::string name = item.get<std::string>();
            if (name != "last_residual" && name != "last_normalized" && name != "top_logits") {
                throw invalid("internal", "Unknown export");
            }
            exports.insert(name);
        }
        const int k = exports.count("top_logits") ? std::max(1, top_k(request)) : 0;
        std::string save_as;
        if (!request.at("save_as").is_null()) {
            save_as = request.at("save_as").get<std::string>();
            require_fresh(save_as);
        }
        const auto started = steady_clock::now();
        branch_result branch;
        try {
            branch = run_branch(parent, full, suffix, save_as);
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            throw protocol_error("execution_error", "", error.what());
        }
        const size_t width = static_cast<size_t>(runtime_.n_embd());
        json vectors_value = json::object();
        if (exports.count("last_residual")) {
            vectors_value["last_residual"] = vector_json(branch.last_residual.data(), 1, width,
                                                         final_representation());
        }
        if (exports.count("last_normalized")) {
            vectors_value["last_normalized"] = vector_json(branch.last_normalized.data(), 1, width,
                                                           "post_final_norm_head_input");
        }
        const readout value = make_readout(branch.vocabulary, {}, k);
        json response = branch.accounting;
        response["type"] = "state";
        response["prompt_sha256"] = sha256_hex(full.prompt);
        response["prompt_tokens"] = full.tokens.size();
        response["timing_ms"] = elapsed_ms(started);
        response["vectors"] = vectors_value;
        response["top_logits"] = k > 0 ? top_json(value) : json::array();
        response["generated_tokens"] = 0;
        return response;
    }

    json inspect(const json & request) {
        const snapshot & item = find(request.at("snapshot_id").get<std::string>());
        json row = snapshot_row(item);
        row["type"] = "snapshot";
        row["token_ids"] = item.tokens;
        row["resident"] = runtime_.resident() == item.id;
        row["holds"] = {
            {"h18", item.h18 != nullptr},
            {"h30", item.h30 != nullptr},
            {"last_normalized", !item.last_normalized.empty()},
            {"upper_kv", item.upper_kv != nullptr},
        };
        return row;
    }

    json vectors(const json & request) {
        const snapshot & item = find(request.at("snapshot_id").get<std::string>());
        const std::string which = request.at("which").get<std::string>();
        const size_t width = static_cast<size_t>(runtime_.n_embd());
        const float * data = nullptr;
        size_t rows = 0;
        std::string representation;
        if (which == "h18" && item.h18) {
            data = item.h18->data();
            rows = item.h18->size() / width;
            representation = "raw_residual_after_block_18";
        } else if (which == "h30" && item.h30) {
            data = item.h30->data();
            rows = item.h30->size() / width;
            representation = final_representation();
        } else if (which == "last_normalized" && !item.last_normalized.empty()) {
            data = item.last_normalized.data();
            rows = 1;
            representation = "post_final_norm_head_input";
        } else {
            throw protocol_error("capability_unavailable", "", "Snapshot does not hold " + which);
        }
        const size_t begin = request.at("row_begin").get<size_t>();
        const size_t end = request.at("row_end").get<size_t>();
        if (begin >= end || end > rows || end - begin > MAX_VECTOR_ROWS) {
            throw invalid("internal", "Row range out of bounds");
        }
        json response = {
            {"type", "vectors"},
            {"snapshot_id", item.id},
            {"which", which},
            {"rows", rows},
            {"row_begin", begin},
            {"row_end", end},
        };
        response["tensor"] = vector_json(data + begin * width, end - begin, width, representation);
        if (end - begin > 1 || which != "last_normalized") {
            response["tensor"]["shape"] = json::array({end - begin, width});
        }
        return response;
    }

    json drop(const json & request) {
        const std::string id = request.at("snapshot_id").get<std::string>();
        const snapshot & item = find(id);
        const size_t freed = snapshot_row(item)["bytes"]["lower_kv"].get<size_t>() +
            snapshot_row(item)["bytes"]["upper_kv"].get<size_t>() +
            snapshot_row(item)["bytes"]["h18"].get<size_t>() +
            snapshot_row(item)["bytes"]["h30"].get<size_t>();
        if (runtime_.resident() == id) runtime_.clear();
        snapshots_.erase(id);
        return {{"type", "dropped"}, {"snapshot_id", id}, {"freed_bytes", freed}};
    }

    // -- persistence -----------------------------------------------------

    static void write_file(const fs::path & path, const uint8_t * data, size_t size) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("Cannot create " + path.string());
        size_t written = 0;
        while (written < size) {
            const ssize_t step = ::write(fd, data + written, size - written);
            if (step <= 0) {
                ::close(fd);
                throw std::runtime_error("Short write to " + path.string());
            }
            written += static_cast<size_t>(step);
        }
        if (::fsync(fd) != 0 || ::close(fd) != 0) throw std::runtime_error("Cannot sync " + path.string());
    }

    static std::vector<uint8_t> read_file(const fs::path & path, size_t limit) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw protocol_error("integrity_error", "", "Missing " + path.filename().string());
        stream.seekg(0, std::ios::end);
        const auto size = static_cast<size_t>(stream.tellg());
        if (size > limit) throw protocol_error("integrity_error", "", "Oversized " + path.filename().string());
        stream.seekg(0);
        std::vector<uint8_t> data(size);
        if (size && !stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size))) {
            throw protocol_error("integrity_error", "", "Cannot read " + path.filename().string());
        }
        return data;
    }

    json save(const json & request) {
        const snapshot & item = find(request.at("snapshot_id").get<std::string>());
        const fs::path directory(request.at("directory").get<std::string>());
        if (!directory.is_absolute() || !fs::is_directory(directory) || !fs::is_empty(directory)) {
            throw invalid("internal", "save needs an existing empty absolute directory");
        }
        json files = json::object();
        const auto put = [&](const std::string & name, const uint8_t * data, size_t size) {
            write_file(directory / name, data, size);
            files[name] = {{"bytes", size}, {"sha256", sha256_hex(data, size)}};
        };
        const auto put_matrix = [&](const std::string & name, const std::vector<float> & values) {
            put(name, reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(float));
        };
        put("lower_kv.bin", item.lower_kv->data(), item.lower_kv->size());
        if (item.upper_kv) put("upper_kv.bin", item.upper_kv->data(), item.upper_kv->size());
        if (item.h18) put_matrix("h18.f32", *item.h18);
        if (item.h30) put_matrix("h30.f32", *item.h30);
        if (!item.last_normalized.empty()) put_matrix("last_normalized.f32", item.last_normalized);
        const json native = {
            {"schema_version", NATIVE_SCHEMA_VERSION},
            {"protocol", PROTOCOL},
            {"profile", config_.profile()},
            {"model_sha256", config_.model_sha256},
            {"runtime_sha256", config_.runtime_sha256},
            {"context_size", config_.context},
            {"batch_size", config_.batch},
            {"n_embd", runtime_.n_embd()},
            {"completed_blocks", item.completed_blocks},
            {"kind", item.kind},
            {"parent", item.parent.empty() ? json() : json(item.parent)},
            {"prompt_sha256", item.prompt_sha256},
            {"token_ids", item.tokens},
            {"files", files},
        };
        const std::string text = native.dump(2) + "\n";
        put("native.json", reinterpret_cast<const uint8_t *>(text.data()), text.size());
        const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0 || ::fsync(fd) != 0) {
            if (fd >= 0) ::close(fd);
            throw std::runtime_error("Cannot sync snapshot directory");
        }
        ::close(fd);
        return {{"type", "saved"}, {"snapshot_id", item.id}, {"files", files}};
    }

    json load(const json & request) {
        const std::string id = request.at("snapshot_id").get<std::string>();
        require_fresh(id);
        const fs::path directory(request.at("directory").get<std::string>());
        if (!directory.is_absolute() || !fs::is_directory(directory)) {
            throw invalid("internal", "load needs an absolute directory");
        }
        const auto & expected = request.at("files");
        if (!expected.is_object() || !expected.contains("native.json")) {
            throw protocol_error("integrity_error", "", "files must name native.json");
        }
        const std::set<std::string> allowed{
            "native.json", "lower_kv.bin", "upper_kv.bin", "h18.f32", "h30.f32", "last_normalized.f32"};
        // Worst case K/V plus matrices for this context; anything bigger is not ours.
        const size_t limit = size_t{4} * 1024 * 1024 * 1024;
        std::map<std::string, std::vector<uint8_t>> data;
        for (const auto & [name, sha] : expected.items()) {
            if (!allowed.count(name)) throw protocol_error("integrity_error", "", "Unexpected file " + name);
            auto bytes = read_file(directory / name, limit);
            if (sha256_hex(bytes.data(), bytes.size()) != sha.get<std::string>()) {
                throw protocol_error("integrity_error", "", "Checksum differs for " + name);
            }
            data.emplace(name, std::move(bytes));
        }
        json native;
        try {
            const auto & text = data.at("native.json");
            native = json::parse(std::string(text.begin(), text.end()));
        } catch (const std::exception &) {
            throw protocol_error("integrity_error", "", "native.json is malformed");
        }
        const auto require = [&](bool condition, const std::string & what) {
            if (!condition) throw protocol_error("integrity_error", "", what);
        };
        require(native.value("schema_version", 0) == NATIVE_SCHEMA_VERSION, "native schema differs");
        require(native.value("profile", "") == config_.profile(), "profile differs");
        require(native.value("model_sha256", "") == config_.model_sha256, "model differs");
        require(native.value("runtime_sha256", "") == config_.runtime_sha256, "runtime differs");
        require(native.value("context_size", 0) == config_.context, "context size differs");
        require(native.value("batch_size", 0) == config_.batch, "batch size differs");
        require(native.value("n_embd", 0) == runtime_.n_embd(), "hidden width differs");
        const int blocks = native.value("completed_blocks", 0);
        const bool whole = runtime_.full();
        const int final_block = runtime_.n_layer();
        require(blocks == final_block || (!whole && blocks == SPLIT_BLOCK), "block coverage differs");
        const auto & listed = native.at("files");
        for (const auto & [name, entry] : listed.items()) {
            require(data.count(name) > 0, "file not verified: " + name);
            require(entry.at("bytes").get<size_t>() == data.at(name).size(), "size differs: " + name);
            require(entry.at("sha256").get<std::string>() == expected.at(name).get<std::string>(),
                    "manifest hash differs: " + name);
        }
        snapshot item;
        item.id = id;
        item.completed_blocks = blocks;
        item.kind = native.value("kind", "");
        require(item.kind == "context" || item.kind == "readout", "kind differs");
        item.parent = native.at("parent").is_null() ? "" : native.at("parent").get<std::string>();
        item.prompt_sha256 = native.value("prompt_sha256", "");
        item.tokens = native.at("token_ids").get<std::vector<llama_token>>();
        const size_t n = item.tokens.size();
        require(n >= 1 && n <= static_cast<size_t>(config_.context), "token count out of range");
        const int vocabulary = llama_vocab_n_tokens(runtime_.vocab());
        for (const auto token : item.tokens) require(token >= 0 && token < vocabulary, "token id out of range");
        const size_t width = static_cast<size_t>(runtime_.n_embd());
        const auto take_matrix = [&](const std::string & name, size_t rows) -> std::shared_ptr<const matrix> {
            if (!data.count(name)) return nullptr;
            const auto & bytes = data.at(name);
            require(bytes.size() == rows * width * sizeof(float), "shape differs: " + name);
            matrix values(rows * width);
            std::memcpy(values.data(), bytes.data(), bytes.size());
            for (const float value : values) require(std::isfinite(value), "nonfinite values: " + name);
            return std::make_shared<const matrix>(std::move(values));
        };
        require(data.count("lower_kv.bin") > 0, "lower K/V missing");
        item.lower_kv = std::make_shared<const blob>(data.at("lower_kv.bin"));
        if (whole) {
            require(data.count("upper_kv.bin") == 0 && data.count("h18.f32") == 0,
                    "a full-depth snapshot holds split state");
        } else if (blocks == final_block) {
            require(data.count("upper_kv.bin") > 0, "upper K/V missing");
            item.upper_kv = std::make_shared<const blob>(data.at("upper_kv.bin"));
        } else {
            require(data.count("upper_kv.bin") == 0, "an 18 snapshot holds upper K/V");
            require(data.count("h18.f32") > 0, "an 18 snapshot needs H18");
        }
        item.h18 = take_matrix("h18.f32", n);
        item.h30 = take_matrix("h30.f32", n);
        if (data.count("last_normalized.f32")) {
            auto last = take_matrix("last_normalized.f32", 1);
            item.last_normalized = *last;
        }
        const json row = snapshot_row(item);
        snapshots_.emplace(id, std::move(item));
        json response = row;
        response["type"] = "loaded";
        return response;
    }

    json reference(const json & request) {
        if (!runtime_.has_reference()) {
            throw protocol_error("capability_unavailable", "", "Worker started without --reference");
        }
        const rendered full = render(request.at("messages"), request.at("answer_prefix").get<std::string>());
        const auto label_ids = label_ids_for(request.at("labels"), full);
        const auto started = steady_clock::now();
        matrix h18;
        matrix h30;
        std::vector<double> vocabulary;
        try {
            runtime_.run_reference(full.tokens, h18, h30, vocabulary);
        } catch (const std::exception & error) {
            throw protocol_error("execution_error", "", error.what());
        }
        const size_t width = static_cast<size_t>(runtime_.n_embd());
        const readout value = make_readout(vocabulary, label_ids, top_k(request));
        json response = readout_json(value);
        response["type"] = "reference";
        response["tokens"] = full.tokens.size();
        response["token_ids"] = full.tokens;
        response["prompt_sha256"] = sha256_hex(full.prompt);
        response["timing_ms"] = elapsed_ms(started);
        response["vectors"] = {
            {"h18_last", vector_json(h18.data() + h18.size() - width, 1, width, "raw_residual_after_block_18")},
            {"h30_last", vector_json(h30.data() + h30.size() - width, 1, width, "raw_residual_after_block_30")},
        };
        if (request.contains("rows")) {
            // Bounded full-row comparison for qualification diagnostics.
            const size_t begin = request.at("rows").at(0).get<size_t>();
            const size_t end = request.at("rows").at(1).get<size_t>();
            if (begin >= end || end > full.tokens.size() || end - begin > MAX_VECTOR_ROWS) {
                throw invalid("internal", "Row range out of bounds");
            }
            response["vectors"]["h18_rows"] = vector_json(
                h18.data() + begin * width, end - begin, width, "raw_residual_after_block_18");
            response["vectors"]["h30_rows"] = vector_json(
                h30.data() + begin * width, end - begin, width, "raw_residual_after_block_30");
        }
        response["generated_tokens"] = 0;
        return response;
    }

    // Greedy autoregressive output. "snapshot" continues a resident or
    // restored 30 parent: the suffix and every new token run lower (0-17),
    // hand H18 to upper (18-29) through llama_batch.embd and read the head;
    // the parent prefix is never recomputed and the contexts are trimmed back
    // to it afterwards. "split_prefill" runs the same split graphs from an
    // empty cache and "reference" the stock 30-block graph; both are the
    // comparison baselines.
    json ride(const json & request) {
        const std::string mode = request.at("mode").get<std::string>();
        if (mode != "snapshot" && mode != "split_prefill" && mode != "reference") {
            throw invalid("internal", "Unknown ride mode");
        }
        if (mode != "snapshot" && !runtime_.has_reference()) {
            throw protocol_error("capability_unavailable", "", "Baseline modes need a --reference worker");
        }
        const int max_tokens = request.at("max_tokens").get<int>();
        if (max_tokens < 1 || max_tokens > config_.context) throw invalid("internal", "max_tokens out of range");
        const int k = top_k(request);
        const rendered full = render(request.at("messages"), request.at("answer_prefix").get<std::string>());
        const snapshot * parent = nullptr;
        size_t reused = 0;
        if (mode == "snapshot") {
            parent = &find(request.at("snapshot_id").get<std::string>());
            if (parent->completed_blocks != runtime_.n_layer()) {
                throw protocol_error("capability_unavailable", "", "Ride needs a 30 snapshot; promote first");
            }
            (void) branch_suffix(*parent, full);
            reused = parent->tokens.size();
        }
        if (full.tokens.size() + static_cast<size_t>(max_tokens) > static_cast<size_t>(config_.context)) {
            throw invalid("budget", "Prompt plus max_tokens exceeds the context");
        }
        // Qualification only: raw f32 logits per step, for exact cross-mode diffs.
        std::ofstream dump;
        if (request.contains("logits_dump") && !request.at("logits_dump").is_null()) {
            if (!runtime_.has_reference()) {
                throw protocol_error("capability_unavailable", "", "logits_dump needs a --reference worker");
            }
            dump.open(request.at("logits_dump").get<std::string>(), std::ios::binary | std::ios::trunc);
            if (!dump) throw invalid("internal", "Cannot open logits_dump");
        }

        std::vector<llama_token> sequence = full.tokens;
        std::vector<double> vocabulary;
        std::vector<float> last_normalized;
        const llama_vocab * vocab = runtime_.vocab();
        size_t restored = 0;
        double restore_ms = 0.0;
        size_t block_tokens = 0;
        const auto started = steady_clock::now();
        // One step through whatever graph the mode uses, for tokens [begin, end).
        const auto advance = [&](size_t begin, size_t end) {
            if (mode == "reference") {
                runtime_.reference_append(sequence, begin, end, begin == 0, vocabulary);
            } else if (runtime_.full()) {
                (void) runtime_.run_full(sequence, begin, end, last_normalized, &vocabulary);
            } else {
                const matrix h18 = runtime_.run_lower(sequence, begin, end);
                (void) runtime_.run_upper(h18, static_cast<llama_pos>(begin), last_normalized, &vocabulary);
            }
            block_tokens += end - begin;
        };
        json steps = json::array();
        std::vector<llama_token> output;
        std::string stop_reason = "max_tokens";
        double ttft_ms = 0.0;
        double decode_ms = 0.0;
        try {
            if (mode == "snapshot") {
                const auto mark = steady_clock::now();
                restored = runtime_.make_resident(*parent);
                restore_ms = elapsed_ms(mark);
            } else if (mode == "split_prefill") {
                runtime_.clear();
            }
            // Baselines may break the prefill where a snapshot would have, so
            // their chunk shapes match the snapshot path exactly.
            const size_t prefill_break = mode == "snapshot" ? 0 : request.value("prefill_break", size_t{0});
            if (prefill_break >= sequence.size()) throw invalid("internal", "prefill_break out of range");
            if (prefill_break > 0) advance(0, prefill_break);
            advance(prefill_break > 0 ? prefill_break : reused, sequence.size());
            ttft_ms = elapsed_ms(started);
            const auto decode_started = steady_clock::now();
            for (;;) {
                const readout value = make_readout(vocabulary, {}, k);
                const llama_token token = static_cast<llama_token>(value.argmax_token);
                if (dump) {
                    const std::vector<float> row(vocabulary.begin(), vocabulary.end());
                    dump.write(reinterpret_cast<const char *>(row.data()),
                               static_cast<std::streamsize>(row.size() * sizeof(float)));
                }
                if (k > 0) steps.push_back({{"token_id", token}, {"top_logits", top_json(value)}});
                if (llama_vocab_is_eog(vocab, token)) {
                    stop_reason = "eog";
                    break;
                }
                output.push_back(token);
                if (static_cast<int>(output.size()) == max_tokens) break;
                sequence.push_back(token);
                advance(sequence.size() - 1, sequence.size());
            }
            decode_ms = elapsed_ms(decode_started);
            if (mode == "snapshot") runtime_.trim_to(*parent);
            else if (mode == "split_prefill") runtime_.clear();
            else runtime_.clear_reference();
        } catch (const protocol_error &) {
            throw;
        } catch (const std::exception & error) {
            runtime_.clear();
            runtime_.clear_reference();
            throw protocol_error("execution_error", "", error.what());
        }
        std::string text;
        for (const llama_token token : output) {
            char piece[256];
            const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
            if (n < 0) throw protocol_error("execution_error", "", "Token piece too long");
            text.append(piece, static_cast<size_t>(n));
        }
        // The first sampled token needs no extra decode; each later one does.
        const size_t decode_steps = output.empty() ? 0 : output.size() - 1 + (stop_reason == "eog" ? 1 : 0);
        return {
            {"type", "ride_result"},
            {"mode", mode},
            {"execution_mode", mode == "reference" ? "stock30" : config_.execution_mode()},
            {"text", text},
            {"token_ids", output},
            {"stop_reason", stop_reason},
            {"prompt_sha256", sha256_hex(full.prompt)},
            {"prompt_tokens", full.tokens.size()},
            {"reused_tokens", reused},
            {"prefilled_tokens", full.tokens.size() - reused},
            {"generated_tokens", output.size()},
            {"block_tokens", mode == "reference" ? json({{"stock", block_tokens}})
                                                 : json({{"lower", block_tokens},
                                                         {"upper", runtime_.full() ? 0 : block_tokens}})},
            {"restore", mode != "snapshot" ? json() : json(restored == 0 ? "resident" : "host")},
            {"restored_bytes", restored},
            {"timing_ms", {
                {"restore", restore_ms},
                {"time_to_first_token", ttft_ms},
                {"decode", decode_ms},
                {"total", elapsed_ms(started)},
            }},
            {"decode_tokens_per_second", decode_ms > 0 ? decode_steps * 1000.0 / decode_ms : 0.0},
            {"steps", steps},
        };
    }

    const settings & config_;
    runtime runtime_;
    const common_chat_templates * templates_;
    std::vector<std::string> labels_;
    std::vector<int> label_ids_;
    std::map<std::string, snapshot> snapshots_;
};

std::pair<std::vector<std::string>, std::vector<int>> validate_alphabet(
        const llama_vocab * vocab, const common_chat_templates * templates) {
    const json messages = json::array({
        {{"role", "user"}, {"content",
            "Choose one option.\n\nOPTIONS:\nA: first\nB: second\n\nReply with one option label only."}},
    });
    const std::string prompt = render_prompt(messages, templates, "Answer:\n");
    const auto prompt_tokens = tokenize(vocab, prompt);
    std::vector<std::string> labels;
    std::vector<int> token_ids;
    for (const char candidate : std::string(LABEL_CANDIDATES)) {
        const std::string label(1, candidate);
        try {
            const auto extended = tokenize(vocab, prompt + label);
            const int token_id = unridden::single_suffix_token(prompt_tokens, extended);
            if (std::find(token_ids.begin(), token_ids.end(), token_id) != token_ids.end()) break;
            labels.push_back(label);
            token_ids.push_back(token_id);
        } catch (const std::runtime_error &) {
            break;
        }
    }
    if (labels.size() < 10) throw std::runtime_error("Validated label alphabet is too small");
    return {labels, token_ids};
}

std::string meta(const llama_model * model, const char * key) {
    char value[256];
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) return "";
    return value;
}

// The split is only sound when no state crosses it other than the residual.
// full-v1 runs the stock graph and needs only a Gemma 4.
void require_profile_compatible(const llama_model * model, bool full) {
    if (meta(model, "general.architecture") != "gemma4") throw std::runtime_error("Worker supports Gemma4 only");
    if (full) return;
    if (llama_model_n_layer(model) != EXPECTED_LAYERS) throw std::runtime_error("Profile needs 30 blocks");
    const std::string shared = meta(model, "gemma4.attention.shared_kv_layers");
    if (!shared.empty() && shared != "0") throw std::runtime_error("Shared KV layers cross the split");
    const std::string per_layer = meta(model, "gemma4.embedding_length_per_layer_input");
    if (!per_layer.empty() && per_layer != "0") throw std::runtime_error("Per-layer inputs cross the split");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const settings config = parse_args(argc, argv);
        reject_incompatible_environment();
        ggml_backend_load_all_from_path(config.runtime_dir.c_str());
        if (config.gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            throw std::runtime_error("GPU opt-in requested but no GPU backend is available");
        }
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.gpu ? 999 : 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(config.model_path.c_str(), model_params), llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        require_profile_compatible(model.get(), config.full);
        char description[256];
        if (llama_model_desc(model.get(), description, sizeof(description)) < 1) {
            throw std::runtime_error("Model identity is unavailable");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        auto templates = common_chat_templates_init(model.get(), "");
        auto [labels, token_ids] = validate_alphabet(vocab, templates.get());
        auto service = std::make_unique<worker>(config, model.get(), templates.get(), labels, token_ids);

        const json hello = {
            {"type", "hello"},
            {"protocol", PROTOCOL},
            {"profile", config.profile()},
            {"model_id", MODEL_ID},
            {"model_name", std::string(description)},
            {"model_sha256", config.model_sha256},
            {"runtime_sha256", config.runtime_sha256},
            {"labels", labels},
            {"label_token_ids", token_ids},
            {"context_size", config.context},
            {"batch_size", config.batch},
            {"ubatch_size", config.ubatch},
            {"threads", config.threads},
            {"n_layer", llama_model_n_layer(model.get())},
            {"n_embd", llama_model_n_embd(model.get())},
            {"split_block", config.full ? json() : json(SPLIT_BLOCK)},
            {"reference_context", config.reference},
            {"context_prompt_version", CONTEXT_PROMPT_VERSION},
            {"generated_tokens", 0},
            {"rider_mode", true},
            {"callbacks_enabled", false},
        };
        std::cout << hello.dump() << '\n' << std::flush;

        std::string line;
        while (read_bounded_line(std::cin, line, MAX_PROTOCOL_BYTES)) {
            if (line.empty()) continue;
            std::string correlation_id = "unknown";
            json response;
            try {
                const json request = json::parse(line);
                correlation_id = request.at("id").get<std::string>();
                if (correlation_id.empty()) throw invalid("internal", "Empty correlation id");
                response = service->handle(request);
            } catch (const protocol_error & error) {
                std::cerr << "REQUEST_FAILED " << error.code << ' ' << error.what() << '\n';
                response = {
                    {"type", "error"},
                    {"code", error.code},
                    {"reason", error.reason.empty() ? json() : json(error.reason)},
                    {"message", error.what()},
                };
            } catch (const std::exception & error) {
                // Malformed envelope or field: the request never reached inference.
                std::cerr << "REQUEST_FAILED invalid_request " << error.what() << '\n';
                response = {
                    {"type", "error"},
                    {"code", "invalid_request"},
                    {"reason", "internal"},
                    {"message", error.what()},
                };
            }
            response["id"] = correlation_id;
            const std::string serialized = response.dump();
            if (serialized.size() > MAX_PROTOCOL_BYTES) throw std::runtime_error("Protocol response exceeds size limit");
            std::cout << serialized << '\n' << std::flush;
        }
        service.reset();
        templates.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "WORKER_FAILED " << error.what() << '\n';
        return 1;
    }
}
