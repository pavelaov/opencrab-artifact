// OpenCrab v2 C++17 mirror: tiny decoder-only transformer executed with the
// Party gadgets (the "tiny_decoder" test model), so that the full
// session loop runs natively in C++ with random or exported weights.
//
// Architecture (pre-LN GPT-2 block): x = wte[token] + wpe[pos]; per layer
// h = LN1(x); qkv = h Wqkv + b; causal multi-head attention over the A64-shared
// KV cache; x += (attn Wout + b); h = LN2(x); x += (gelu(h Wfc + bfc) Wproj + bproj);
// logits = LNf(x_last) Wlm; token = argmax (released to P1 only).
//
// Plans ({"embedding": wmask|generic, "linear": wmask|generic, "argmax_k": 2|4|8,
// "max_k": 2|4}) select the realization of the embedding site, of the four
// linear sites per layer, and the tournament arities. The LM head is wmask
// under every plan (its install is part of the cold-start setup, as in the
// cost model, costmodel.hpp). Installed wmask objects: "wte" (vocab x d),
// "h<l>.qkv" (d x 3d), "h<l>.out" (d x d), "h<l>.fc" (d x d_ff),
// "h<l>.proj" (d_ff x d), "lm_head" (d x vocab).
//
// Weight file format (little-endian), so that weights exported from Python
// (numpy float32) can be loaded:
//   magic[8] = "OC2TDW01"
//   u32 n_layers, u32 d_model, u32 n_head, u32 vocab, u32 max_pos, u32 d_ff
//   float32 arrays, row-major, in this order:
//     wte[vocab][d_model], wpe[max_pos][d_model],
//     for each layer: ln1_g[d], ln1_b[d], w_qkv[d][3d], b_qkv[3d], w_out[d][d], b_out[d],
//                     ln2_g[d], ln2_b[d], w_fc[d][d_ff], b_fc[d_ff], w_proj[d_ff][d], b_proj[d],
//     lnf_g[d], lnf_b[d], w_lm[d][vocab]
// TinyWeights::random generates a deterministic random model for tests.
//
// Server-private vectors (biases, LN parameters) and the weight matrices of the
// generic realization are shared with Party::share_input at setup (one message
// each, charged to T_cold); the dealer never receives weights.
#pragma once

#include <opencrab2/highlevel/party.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace opencrab2 {
namespace highlevel {

struct TinyConfig {
    uint32_t n_layers = 1;
    uint32_t d_model = 16;
    uint32_t n_head = 2;
    uint32_t vocab = 32;
    uint32_t max_pos = 64;
    uint32_t d_ff = 64;
    Value to_value() const;
};

struct TinyLayer {
    std::vector<float> ln1_g, ln1_b, w_qkv, b_qkv, w_out, b_out, ln2_g, ln2_b, w_fc, b_fc, w_proj, b_proj;
};

struct TinyWeights {
    TinyConfig cfg;
    std::vector<float> wte, wpe;
    std::vector<TinyLayer> layers;
    std::vector<float> lnf_g, lnf_b, w_lm;

    static TinyWeights random(const TinyConfig& cfg, uint64_t seed, float std_dev = 0.02f, float emb_std = 0.3f);
    Bytes serialize() const;
    static Expected<TinyWeights> deserialize(const Bytes& b);
    Expected<Unit> save(const std::string& path) const;
    static Expected<TinyWeights> load(const std::string& path);
    std::string digest() const;  // sha256 hex of serialize(): the model_version
};

struct TinyPlan {
    std::string embedding = "generic";
    std::string linear = "generic";
    int argmax_k = 2;
    int max_k = 2;
    Json to_json() const;
    static Expected<TinyPlan> from_json(const Json& j);
    std::string key() const;
};

// Plaintext (double) reference of the same network, greedy decoding.
class TinyReference {
public:
    explicit TinyReference(const TinyWeights& w) : w_(w) {}
    // Returns the n_generate greedily decoded tokens after the prompt.
    std::vector<int> generate(const std::vector<int>& prompt, int n_generate) const;
    // Logits of the last position for the given full token sequence.
    std::vector<double> last_logits(const std::vector<int>& tokens) const;

private:
    const TinyWeights& w_;
};

class TinyDecoder {
public:
    // P0 passes its weights, P1 nullptr; both pass the public config.
    TinyDecoder(Party& party, const TinyWeights* weights_or_null, const TinyConfig& cfg, TinyPlan plan);

    // Shares the server-private vectors and matrices and installs the LM head
    // (charged to T_cold). Returns {bytes, seconds, rounds}.
    Expected<Value> setup();
    // Installs every wmask object the plan needs that is not installed yet.
    // Returns {objects: [...], bytes, seconds, rounds, count}.
    Expected<Value> install_for_plan(const TinyPlan& plan);
    void set_plan(const TinyPlan& plan) { plan_ = plan; }
    const TinyPlan& plan() const { return plan_; }
    std::vector<std::string> required_objects(const TinyPlan& plan) const;
    std::vector<std::string> missing_objects(const TinyPlan& plan) const;

    // P1 passes its token ids (n of them), P0 nullptr. Returns the released
    // token on P1 and -1 on P0.
    Expected<int> prefill(const std::vector<int>* token_ids, size_t n);
    Expected<int> decode_step(int token_id_or_minus1, size_t position);

    size_t kv_len() const { return T_; }
    Value installed_objects() const { return p_.installed_objects(); }
    const TinyConfig& config() const { return cfg_; }
    void reset_cache();

private:
    Expected<Mat> embed(const std::vector<int>* ids, size_t n, size_t pos0);
    Expected<Mat> linear(const Mat& x, const std::string& mat_id);
    Expected<Mat> layer(size_t li, const Mat& x);
    Expected<Mat> attention(size_t li, const Mat& h);
    Expected<int> head(const Mat& x_last);
    Expected<Unit> share_vector(const std::string& name, const std::vector<float>* v, size_t rows, size_t cols);

    Party& p_;
    const TinyWeights* W_;
    TinyConfig cfg_;
    TinyPlan plan_;
    std::vector<Mat> K_, V_;  // per layer, T x d
    size_t T_ = 0;
    bool setup_done_ = false;
    std::map<std::string, Mat> enc_mats_;  // P0: encoded weight matrices by object id
};

}  // namespace highlevel
}  // namespace opencrab2
