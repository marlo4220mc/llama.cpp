#pragma once

#include "ggml-backend.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct ggml_tensor;

enum class llama_expert_fill_status {
    LOADED,            // Experto transferido al buffer de VRAM exitosamente
    FALLBACK_ALLOWED,  // Tensor excluido de streaming (permitido leer de w->data)
    NOT_SUPPORTED,     // Layout/geometría no compatible con streaming
    IO_ERROR,          // Error físico de lectura o pread corrupto
    INVALID_EXTENT,    // Desbordamiento de límites de archivo o metadatos inválidos
    OUT_OF_MEMORY      // Memoria insuficiente para L2 o staging
};

struct llama_expert_stats {
    std::atomic<uint64_t> fills           { 0 };
    std::atomic<uint64_t> l2_hits         { 0 };
    std::atomic<uint64_t> l2_misses       { 0 };
    std::atomic<uint64_t> proc_bytes      { 0 };
    std::atomic<uint64_t> io_us_total     { 0 };
    std::atomic<uint64_t> io_us_max       { 0 };
    std::atomic<uint64_t> drops           { 0 };
    std::atomic<uint64_t> verify_ok       { 0 };
    std::atomic<uint64_t> verify_fail     { 0 };

    void log(const char * tag) const;
    void reset();
};

struct llama_expert_file_meta {
    std::string path;
    uint64_t    size = 0;
};

struct llama_expert_tensor_meta {
    int32_t  shard_idx     = -1;
    uint64_t base_offset   = 0;
    uint64_t expert_stride = 0; // t->nb[2] real
    uint64_t expert_bytes  = 0;
    int32_t  n_expert      = 0;
    bool     is_supported  = false;
};

struct llama_expert_model_meta {
    std::vector<llama_expert_file_meta> files;
    std::unordered_map<const ggml_tensor *, llama_expert_tensor_meta> tensors;
};

// Función canónica para lectura robusta desde almacenamiento
int64_t llama_expert_pread_full(int fd, void * buf, size_t count, uint64_t offset, uint64_t file_size);

class llama_expert_source {
public:
    struct config {
        size_t ring_bytes     = 64ull * 1024 * 1024;
        size_t ram_warm_bytes = 1024ull * 1024 * 1024;
        bool   stats          = false;
        bool   verify         = false;
    };

    static std::unique_ptr<llama_expert_source> create(
        const llama_expert_model_meta & meta,
        const config & cfg);

    ~llama_expert_source();

    // Rellena el experto faltante en el banco de VRAM
    llama_expert_fill_status fill_expert(
        ggml_backend_t backend,
        ggml_tensor * bank,
        int32_t il,
        const ggml_tensor * w,
        int32_t expert,
        size_t slot_offset,
        size_t expert_size);

    // Se invoca cuando una capa previa ha sido confirmada como sincronizada en el backend
    void on_layer_synchronized(int32_t il);

    // Sincronización explícita final (para última capa o destrucción)
    void synchronize_all();

    const llama_expert_stats & get_stats() const;

    // Métodos para inspección y pruebas unitarias
    size_t get_l2_capacity_pairs() const;
    size_t get_staging_ring_slots() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
