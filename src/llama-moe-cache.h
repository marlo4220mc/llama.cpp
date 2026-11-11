#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // n_threads is the number of CPU threads that compute the misses of small batches within the graph
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, int32_t n_threads);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // batches of up to 8 tokens compute the misses within the graph, without slot lookups (see ggml-moe-cpu-misses.h):
    // the device looks up the slots, the CPU computes a share of the misses, and the other misses are cached by the device or between the graphs
    // can_cpu_part: the graph can compute the misses of layer il on the CPU
    bool single(int32_t il, int64_t n_tokens, bool can_cpu_part) const;

    // the graphs of single() merge the result of a CPU part, see set_cpu_graph
    bool single_cpu_part() const;

    // small ubatches (the backend sets the limit for the expert types of layer il) can compute the hits while the device copies the misses,
    // then the misses: the publish op then gives two slot planes [n_used, n_tokens, 2], the hits and the zero slots of the experts of the CPU
    // (-1 for the misses the device caches) and the misses the device caches (-1 for the others), build_join gives the second plane once
    // the misses are copied, and build_merge adds the result of the CPU after both passes
    bool two_pass(int32_t il, int64_t n_tokens) const;

    // the slots of the experts selected in layer il, the misses of the CPU read zero slots
    ggml_tensor * build_publish(ggml_context * ctx, ggml_tensor * selected_experts, ggml_tensor * cur, int32_t il, bool two_pass) const;
    ggml_tensor * build_join(ggml_context * ctx, ggml_tensor * slots, int32_t il) const;

    // the result of the cached experts of layer il plus the result of the CPU part
    ggml_tensor * build_merge(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * slots, int32_t il) const;

    // the CPU part of layer il for n_tokens tokens: input F32 [n_embd, 1, n_tokens], selected experts and the experts of the CPU I32 [n_used, n_tokens],
    // result F32 [n_embd, n_used, n_tokens], the graph of the result is copied and the tensors are not used after the call
    bool has_cpu_graph(int32_t il, int64_t n_tokens) const;
    void set_cpu_graph(int32_t il, ggml_tensor * input, ggml_tensor * selected, ggml_tensor * ids, ggml_tensor * result);

    // aborts if the graph has both publish ops and slot lookups, before the scheduler allocates it
    // publish: (layer, experts per token, tokens) of each publish op of the graph, for begin()
    void check_graph(ggml_cgraph * graph, std::vector<int32_t> & publish) const;

    // the thread that queues the graph can compute the CPU parts: no CPU op runs after the first publish op
    bool main_ok(ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_backend_t cpu) const;

    // called before each graph compute, returns false if the CPU part of the last graph failed
    // main: the thread that queues the graph computes its CPU parts in serve(), else a CPU thread of the cache
    // publish: as given by check_graph() for this graph, or nullptr to search the graph
    bool begin(ggml_cgraph * graph, bool main, const std::vector<int32_t> * publish = nullptr);

    // called after the graph is queued: the host replays the slots of the last graph, then serves the CPU parts
    void serve(ggml_backend_t cpu);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
