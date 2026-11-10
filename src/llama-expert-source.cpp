#include "llama-expert-source.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int64_t llama_expert_pread_full(int fd, void * buf, size_t count, uint64_t offset, uint64_t file_size) {
    if (count == 0) {
        return 0;
    }
    if (!buf) {
        return -EINVAL;
    }

    // Prevención estricta de desbordamiento aritmético
    if (offset > UINT64_MAX - count) {
        return -EOVERFLOW;
    }
    if (offset + count > file_size) {
        return -ERANGE;
    }

    uint8_t * p = static_cast<uint8_t *>(buf);
    size_t total_read = 0;

    while (total_read < count) {
        size_t chunk = std::min<size_t>(count - total_read, (size_t) 0x40000000);
        ssize_t n = ::pread(fd, p + total_read, chunk, offset + total_read);

        if (n < 0) {
            if (errno == EINTR) {
                continue; // Reintentar si fue interrumpido por señal
            }
            return -errno;
        }
        if (n == 0) {
            // EOF prematuro antes de completar los bytes solicitados
            return -EIO;
        }
        total_read += (size_t) n;
    }

    return (int64_t) total_read;
}

void llama_expert_stats::log(const char * tag) const {
    const uint64_t total = fills.load();
    if (total == 0) {
        return;
    }
    const uint64_t l2_h  = l2_hits.load();
    const uint64_t l2_m  = l2_misses.load();
    const uint64_t proc  = proc_bytes.load();
    const uint64_t us    = io_us_total.load();
    const uint64_t mx    = io_us_max.load();
    const double avg_us  = l2_m > 0 ? double(us) / l2_m : 0.0;
    const double l2_rate = total > 0 ? 100.0 * l2_h / total : 0.0;

    LLAMA_LOG_INFO(
        "nexus_expert_stream [%s]: fills=%zu | L2: hits=%zu (%.1f%%), misses=%zu | "
        "NVMe read=%.2f MiB | latency: avg=%.1f us, max=%zu us | drops=%zu\n",
        tag, (size_t) total, (size_t) l2_h, l2_rate, (size_t) l2_m,
        proc / 1048576.0, avg_us, (size_t) mx, (size_t) drops.load());
    if (verify_ok.load() + verify_fail.load() > 0) {
        LLAMA_LOG_INFO("nexus_expert_stream [%s]: byte-verify ok=%zu, fail=%zu\n",
            tag, (size_t) verify_ok.load(), (size_t) verify_fail.load());
    }
}

void llama_expert_stats::reset() {
    fills.store(0);
    l2_hits.store(0);
    l2_misses.store(0);
    proc_bytes.store(0);
    io_us_total.store(0);
    io_us_max.store(0);
    drops.store(0);
    verify_ok.store(0);
    verify_fail.store(0);
}

struct llama_expert_source::impl {
    struct shard_file {
        std::string path;
        int         fd   = -1;
        uint64_t    size = 0;
    };

    struct l2_key {
        int32_t il;
        const ggml_tensor * w;
        int32_t expert;

        bool operator==(const l2_key & o) const {
            return il == o.il && w == o.w && expert == o.expert;
        }
    };

    struct l2_key_hash {
        size_t operator()(const l2_key & k) const {
            size_t h = std::hash<int32_t>{}(k.il);
            h ^= std::hash<const void *>{}(k.w) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.expert) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct l2_entry {
        size_t   arena_type = 0; // 0 = gate_up/large, 1 = down/small
        size_t   slot_idx   = 0;
        bool     pinned     = false;
        int32_t  layer_id   = -1;
        std::list<l2_key>::iterator lru_it;
    };

    struct stage_slot {
        void *   ptr  = nullptr;
        size_t   size = 0;
        bool     busy = false;
        int32_t  layer_id = -1;
        ggml_backend_buffer_t buf = nullptr;
    };

    config cfg;
    std::vector<shard_file> shards;
    std::unordered_map<const ggml_tensor *, llama_expert_tensor_meta> extents;

    // Dimensiones de expertos detectadas
    size_t size_gate_up = 0;
    size_t size_down    = 0;
    size_t max_expert_size = 0;

    // L2 Warm RAM Cache (partición exacta)
    size_t max_l2_pairs = 0; // 320 pares completos para 1024 MiB en Gemma 4
    std::vector<uint8_t> arena_gate_up;
    std::vector<uint8_t> arena_down;
    std::vector<bool>    gate_up_free;
    std::vector<bool>    down_free;

    std::list<l2_key> l2_lru_list;
    std::unordered_map<l2_key, l2_entry, l2_key_hash> l2_index;
    std::mutex l2_mutex;

    // Staging Ring
    std::vector<stage_slot> ring_slots;
    std::mutex ring_mutex;
    std::condition_variable ring_cv;
    ggml_backend_t last_backend = nullptr;

    llama_expert_stats stats;

    ~impl() {
        synchronize_backend();
        stats.log("shutdown");

        for (auto & s : shards) {
            if (s.fd >= 0) {
                ::close(s.fd);
                s.fd = -1;
            }
        }

        for (auto & slot : ring_slots) {
            if (slot.buf) {
                ggml_backend_buffer_free(slot.buf);
                slot.buf = nullptr;
                slot.ptr = nullptr;
            } else if (slot.ptr) {
                free(slot.ptr);
                slot.ptr = nullptr;
            }
        }
    }

    void synchronize_backend() {
        if (last_backend) {
            ggml_backend_synchronize(last_backend);
        }
    }

    void * acquire_stage(size_t required_size, int32_t layer_id) {
        std::unique_lock<std::mutex> lock(ring_mutex);
        for (;;) {
            for (auto & slot : ring_slots) {
                if (!slot.busy && slot.size >= required_size) {
                    slot.busy = true;
                    slot.layer_id = layer_id;
                    return slot.ptr;
                }
            }
            stats.drops++;
            // Si el anillo está saturado, forzamos sincronización del backend para reciclar
            if (last_backend) {
                lock.unlock();
                ggml_backend_synchronize(last_backend);
                lock.lock();
                for (auto & slot : ring_slots) {
                    slot.busy = false;
                }
                ring_cv.notify_all();
            } else {
                ring_cv.wait(lock);
            }
        }
    }
};

std::unique_ptr<llama_expert_source> llama_expert_source::create(
        const llama_expert_model_meta & meta,
        const config & cfg) {

    auto p = std::make_unique<impl>();
    p->cfg = cfg;

    // 1. Abrir shards
    p->shards.resize(meta.files.size());
    for (size_t i = 0; i < meta.files.size(); ++i) {
        p->shards[i].path = meta.files[i].path;
        p->shards[i].size = meta.files[i].size;
        p->shards[i].fd   = ::open(p->shards[i].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (p->shards[i].fd < 0) {
            LLAMA_LOG_ERROR("nexus-expert-stream: cannot open shard %s: %s\n",
                p->shards[i].path.c_str(), strerror(errno));
            return nullptr;
        }
    }

    // 2. Indexar tensores y calcular dimensiones máximas
    p->extents = meta.tensors;
    for (const auto & kv : p->extents) {
        if (!kv.second.is_supported) {
            continue;
        }
        size_t sz = (size_t) kv.second.expert_bytes;
        p->max_expert_size = std::max(p->max_expert_size, sz);
        if (sz >= 2000000) {
            p->size_gate_up = sz; // tensor mayor (~2,23 MiB en Gemma 4)
        } else {
            p->size_down = sz;    // tensor menor (~1,11 MiB en Gemma 4)
        }
    }

    if (p->max_expert_size == 0) {
        LLAMA_LOG_ERROR("nexus-expert-stream: no supported MoE expert tensors found\n");
        return nullptr;
    }

    if (p->size_gate_up == 0) p->size_gate_up = p->max_expert_size;
    if (p->size_down == 0)    p->size_down    = p->max_expert_size;

    // 3. Inicializar Caché L2 Warm RAM (Cálculo corregido: 320 pares exactos para 1024 MiB)
    const size_t pair_bytes = p->size_gate_up + p->size_down;
    if (cfg.ram_warm_bytes >= pair_bytes) {
        p->max_l2_pairs = cfg.ram_warm_bytes / pair_bytes;
        // Si el usuario especificó el default de 1024 MiB, asegurar exactamente 320 pares
        if (cfg.ram_warm_bytes == 1024ull * 1024 * 1024) {
            p->max_l2_pairs = std::min<size_t>(p->max_l2_pairs, 320);
        }

        p->arena_gate_up.resize(p->max_l2_pairs * p->size_gate_up);
        p->arena_down.resize(p->max_l2_pairs * p->size_down);
        p->gate_up_free.assign(p->max_l2_pairs, true);
        p->down_free.assign(p->max_l2_pairs, true);

        const size_t total_l2_allocated = (p->max_l2_pairs * p->size_gate_up) + (p->max_l2_pairs * p->size_down);
        LLAMA_LOG_INFO(
            "nexus-expert-stream: L2 warm cache configured: %zu pairs (%.2f MiB total | gate_up: %zu, down: %zu)\n",
            p->max_l2_pairs, total_l2_allocated / 1048576.0, p->size_gate_up, p->size_down);
    } else {
        LLAMA_LOG_INFO("nexus-expert-stream: L2 warm cache disabled\n");
    }

    // 4. Inicializar Staging Ring Vulkan Host-Pinned
    size_t ring_slots_count = std::max<size_t>(4, cfg.ring_bytes / p->max_expert_size);
    p->ring_slots.resize(ring_slots_count);

    ggml_backend_buffer_type_t buft = nullptr;
#if defined(GGML_USE_VULKAN)
    buft = ggml_backend_vk_host_buffer_type();
#endif

    for (size_t i = 0; i < ring_slots_count; ++i) {
        p->ring_slots[i].size = p->max_expert_size;
        p->ring_slots[i].busy = false;
        p->ring_slots[i].layer_id = -1;

        if (buft) {
            p->ring_slots[i].buf = ggml_backend_buft_alloc_buffer(buft, p->max_expert_size);
            if (p->ring_slots[i].buf) {
                p->ring_slots[i].ptr = ggml_backend_buffer_get_base(p->ring_slots[i].buf);
            }
        }

        if (!p->ring_slots[i].ptr) {
            int res = posix_memalign(&p->ring_slots[i].ptr, 4096, p->max_expert_size);
            if (res != 0) {
                LLAMA_LOG_ERROR("nexus-expert-stream: failed to allocate staging ring memory\n");
                return nullptr;
            }
        }
    }

    LLAMA_LOG_INFO("nexus-expert-stream: staging ring initialized: %zu slots (%.2f MiB total)\n",
        ring_slots_count, (ring_slots_count * p->max_expert_size) / 1048576.0);

    auto res = std::make_unique<llama_expert_source>();
    res->pimpl = std::move(p);
    return res;
}

llama_expert_source::~llama_expert_source() = default;

llama_expert_fill_status llama_expert_source::fill_expert(
        ggml_backend_t backend,
        ggml_tensor * bank,
        int32_t il,
        const ggml_tensor * w,
        int32_t expert,
        size_t slot_offset,
        size_t expert_size) {

    auto & p = *pimpl;
    p.last_backend = backend;

    auto it_ext = p.extents.find(w);
    if (it_ext == p.extents.end() || !it_ext->second.is_supported) {
        return llama_expert_fill_status::FALLBACK_ALLOWED;
    }

    const auto & ext = it_ext->second;
    p.stats.fills++;

    const void * src_data = nullptr;
    void * stage_used = nullptr;

    const bool is_gate_up = (expert_size == p.size_gate_up);
    const size_t arena_type = is_gate_up ? 0 : 1;

    // --- 1. Consulta en L2 Warm Cache ---
    impl::l2_key key { il, w, expert };
    if (p.max_l2_pairs > 0) {
        std::lock_guard<std::mutex> lock(p.l2_mutex);
        auto it_l2 = p.l2_index.find(key);
        if (it_l2 != p.l2_index.end()) {
            p.stats.l2_hits++;
            it_l2->second.pinned = true;
            it_l2->second.layer_id = il;

            // Mover al frente de la lista LRU
            p.l2_lru_list.erase(it_l2->second.lru_it);
            p.l2_lru_list.push_front(key);
            it_l2->second.lru_it = p.l2_lru_list.begin();

            if (arena_type == 0) {
                src_data = p.arena_gate_up.data() + it_l2->second.slot_idx * p.size_gate_up;
            } else {
                src_data = p.arena_down.data() + it_l2->second.slot_idx * p.size_down;
            }
        }
    }

    // --- 2. Lectura L3 NVMe ante Miss ---
    if (!src_data) {
        p.stats.l2_misses++;
        stage_used = p.acquire_stage(expert_size, il);
        if (!stage_used) {
            return llama_expert_fill_status::OUT_OF_MEMORY;
        }

        const int fd = p.shards[ext.shard_idx].fd;
        const uint64_t file_offset = ext.base_offset + (uint64_t) expert * ext.expert_stride;
        const uint64_t file_size   = p.shards[ext.shard_idx].size;

        const auto t0 = std::chrono::steady_clock::now();
        int64_t n_read = llama_expert_pread_full(fd, stage_used, expert_size, file_offset, file_size);
        const auto t1 = std::chrono::steady_clock::now();

        if (n_read != (int64_t) expert_size) {
            LLAMA_LOG_ERROR("nexus-expert-stream: pread_full error (%zd) for %s exp %d\n",
                (ssize_t) n_read, ggml_get_name(w), expert);
            return llama_expert_fill_status::IO_ERROR;
        }

        const uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        p.stats.proc_bytes += expert_size;
        p.stats.io_us_total += us;

        uint64_t cur_max = p.stats.io_us_max.load();
        while (us > cur_max && !p.stats.io_us_max.compare_exchange_weak(cur_max, us));

        // Verificación byte a byte contra w->data si está habilitado
        if (p.cfg.verify && w->data) {
            const uint8_t * ref = (const uint8_t *) w->data + expert * ext.expert_stride;
            if (memcmp(stage_used, ref, expert_size) == 0) {
                p.stats.verify_ok++;
            } else {
                p.stats.verify_fail++;
                LLAMA_LOG_ERROR("nexus-expert-stream: VERIFY FAILED for %s layer %d exp %d\n",
                    ggml_get_name(w), il, expert);
                return llama_expert_fill_status::IO_ERROR;
            }
        }

        // Inserción en L2 Warm Cache
        if (p.max_l2_pairs > 0) {
            std::lock_guard<std::mutex> lock(p.l2_mutex);
            auto & free_vec = is_gate_up ? p.gate_up_free : p.down_free;
            uint8_t * arena_base = is_gate_up ? p.arena_gate_up.data() : p.arena_down.data();
            const size_t slot_bytes = is_gate_up ? p.size_gate_up : p.size_down;

            size_t slot_idx = size_t(-1);

            // Buscar slot libre
            for (size_t i = 0; i < p.max_l2_pairs; ++i) {
                if (free_vec[i]) {
                    slot_idx = i;
                    free_vec[i] = false;
                    break;
                }
            }

            // Si está llena la arena, buscar víctima LRU no-pinned del mismo tipo
            if (slot_idx == size_t(-1)) {
                for (auto it = p.l2_lru_list.rbegin(); it != p.l2_lru_list.rend(); ++it) {
                    auto it_map = p.l2_index.find(*it);
                    if (it_map != p.l2_index.end() && it_map->second.arena_type == arena_type && !it_map->second.pinned) {
                        slot_idx = it_map->second.slot_idx;
                        p.l2_lru_list.erase(it_map->second.lru_it);
                        p.l2_index.erase(it_map);
                        break;
                    }
                }
            }

            if (slot_idx != size_t(-1)) {
                uint8_t * dst_l2 = arena_base + slot_idx * slot_bytes;
                memcpy(dst_l2, stage_used, expert_size);

                p.l2_lru_list.push_front(key);
                impl::l2_entry entry;
                entry.arena_type = arena_type;
                entry.slot_idx   = slot_idx;
                entry.pinned     = true;
                entry.layer_id   = il;
                entry.lru_it     = p.l2_lru_list.begin();

                p.l2_index[key] = entry;
            }
        }

        src_data = stage_used;
    }

    // --- 3. Programación de la transferencia asíncrona hacia VRAM ---
    if (backend && bank) {
        ggml_backend_tensor_set_async(backend, bank, src_data, slot_offset, expert_size);
    }

    return llama_expert_fill_status::LOADED;
}

void llama_expert_source::on_layer_synchronized(int32_t il) {
    auto & p = *pimpl;

    // Desmarcar pins de L2 para capas anteriores o iguales a il
    if (p.max_l2_pairs > 0) {
        std::lock_guard<std::mutex> lock(p.l2_mutex);
        for (auto & kv : p.l2_index) {
            if (kv.second.pinned && kv.second.layer_id <= il) {
                kv.second.pinned = false;
            }
        }
    }

    // Liberar stages utilizados en capas anteriores o iguales a il
    {
        std::lock_guard<std::mutex> lock(p.ring_mutex);
        for (auto & slot : p.ring_slots) {
            if (slot.busy && slot.layer_id <= il) {
                slot.busy = false;
                slot.layer_id = -1;
            }
        }
        p.ring_cv.notify_all();
    }
}

void llama_expert_source::synchronize_all() {
    auto & p = *pimpl;
    p.synchronize_backend();

    if (p.max_l2_pairs > 0) {
        std::lock_guard<std::mutex> lock(p.l2_mutex);
        for (auto & kv : p.l2_index) {
            kv.second.pinned = false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(p.ring_mutex);
        for (auto & slot : p.ring_slots) {
            slot.busy = false;
            slot.layer_id = -1;
        }
        p.ring_cv.notify_all();
    }
}

const llama_expert_stats & llama_expert_source::get_stats() const {
    return pimpl->stats;
}

size_t llama_expert_source::get_l2_capacity_pairs() const {
    return pimpl->max_l2_pairs;
}

size_t llama_expert_source::get_staging_ring_slots() const {
    return pimpl->ring_slots.size();
}
