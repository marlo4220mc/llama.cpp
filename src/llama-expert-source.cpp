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
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Hash para llama_expert_key
size_t llama_expert_key_hash::operator()(const llama_expert_key & k) const {
    size_t h = std::hash<int32_t>{}(k.layer_id);
    h ^= std::hash<uintptr_t>{}(k.tensor_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int32_t>{}(k.expert_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

// Lectura canónica robusta desde NVMe
int64_t llama_expert_pread_full(int fd, void * buf, size_t count, uint64_t offset, uint64_t file_size) {
    if (count == 0) {
        return 0;
    }
    if (!buf) {
        return -EINVAL;
    }

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
                continue;
            }
            return -errno;
        }
        if (n == 0) {
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
    const uint64_t bytes = proc_bytes.load();
    const uint64_t us    = io_us_total.load();
    const uint64_t max_u = io_us_max.load();
    const double avg_us  = (l2_m > 0) ? (double) us / l2_m : 0.0;
    const double mb      = (double) bytes / (1024.0 * 1024.0);
    const double thr     = (us > 0) ? (mb / ((double) us / 1000000.0)) : 0.0;

    LLAMA_LOG_INFO(
        "nexus-expert-stream [%s]: fills=%lu (L2 hit=%lu miss=%lu | hit_rate=%.1f%%) | NVMe: %.2f MiB, avg=%.1f us, max=%lu us (%.2f MiB/s)\n",
        tag,
        (unsigned long) total,
        (unsigned long) l2_h,
        (unsigned long) l2_m,
        100.0 * (double) l2_h / (double) total,
        mb, avg_us, (unsigned long) max_u, thr);
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

// ---------------------------------------------------------------------------
// Allocator First-Fit & Cuarentena
// ---------------------------------------------------------------------------

namespace {

bool safe_align_up(size_t req_bytes, size_t A, size_t & out_aligned) noexcept {
    if (A == 0 || (A & (A - 1)) != 0) {
        return false;
    }
    if (req_bytes > SIZE_MAX - (A - 1)) {
        return false;
    }
    size_t res = (req_bytes + (A - 1)) & ~(A - 1);
    if (res < req_bytes) {
        return false;
    }
    out_aligned = res;
    return true;
}

enum class fail_reason {
    none,
    epoch_overflow,
    epoch_regression,
    sync_failed,
    teardown_sync_failed,
    inflight_without_block,
    inflight_underflow,
    residual_inflight,
    lifetime_violation
};

struct quarantine_node {
    ggml_backend_buffer_t arena_buf   = nullptr;
    void *                arena_base  = nullptr;
    size_t                arena_bytes = 0;
    bool                  owns_arena  = false;
    ggml_backend_t        backend     = nullptr;
    fail_reason           reason      = fail_reason::none;
    bool                  fail_stop   = false;
    quarantine_node *     next        = nullptr;
};

static_assert(std::is_nothrow_default_constructible<quarantine_node>::value,
    "quarantine_node must be nothrow default constructible");

struct process_arena_quarantine {
    static_assert(
        std::atomic<quarantine_node *>::is_always_lock_free,
        "nexus D-0 requires lock-free pointer atomics for the quarantine list");

    std::atomic<quarantine_node *> head{nullptr};

    void enroll(quarantine_node * node) noexcept {
        if (!node) return;
        quarantine_node * current = head.load(std::memory_order_relaxed);
        do {
            node->next = current;
        } while (!head.compare_exchange_weak(
            current, node,
            std::memory_order_release,
            std::memory_order_relaxed));
    }
};

static process_arena_quarantine g_arena_quarantine;

} // namespace

// Metadatos por bloque vivo
struct nexus_l2_block {
    size_t                offset          = 0;
    size_t                size            = 0;
    size_t                slack           = 0;
    size_t                req_bytes       = 0;
    llama_expert_content  content         = llama_expert_content::EMPTY;
    int32_t               pin_layers      = 0;
    int32_t               inflight        = 0;
    bool                  needs_proof     = false;
    uint64_t              lru_tick        = 0;
    uint64_t              last_fill_epoch = 0;
    std::list<llama_expert_key>::iterator lru_it;
};

struct llama_expert_source::impl {
    struct fill_active_guard;

    struct shard_file {
        std::string path;
        int         fd   = -1;
        uint64_t    size = 0;
    };

    config cfg;
    std::vector<shard_file> shards;
    std::unordered_map<const ggml_tensor *, llama_expert_tensor_meta> extents;

    size_t size_gate_up    = 0;
    size_t size_down       = 0;
    size_t max_expert_size = 0;

    // Memoria física de la arena L2 única
    ggml_backend_buffer_t arena_buf   = nullptr;
    void *                arena_base  = nullptr;
    size_t                arena_bytes = 0;
    size_t                alignment   = 0;
    bool                  owns_arena  = false;
    // la arena se retiro del uso sin liberarla (cuarentena del proceso): puede
    // haber una transferencia del dispositivo que aun la este leyendo
    std::atomic<bool>     arena_quarantined{false};
    llama_expert_mem_kind mem_kind    = llama_expert_mem_kind::PAGEABLE_HOST;

    // Allocator First-Fit
    std::map<size_t, size_t> free_holes; // offset -> size
    size_t aligned_bytes = 0;
    size_t slack_bytes   = 0;
    size_t free_bytes    = 0;

    // Índice de bloques y LRU
    std::unordered_map<llama_expert_key, nexus_l2_block, llama_expert_key_hash> l2_index;
    std::list<llama_expert_key> l2_lru_list;
    uint64_t current_lru_tick = 0;

    // Concurrencia y Barrera
    mutable std::mutex l2_mutex;
    std::condition_variable cv_idle;
    std::condition_variable cv_gate;
    std::condition_variable cv_quiescent;
    std::condition_variable cv_loading;

    bool     gate_closed       = false;
    uint64_t gate_epoch        = 0;
    int      programming_fills = 0;
    int      active_fills      = 0;
    bool     barrier_active    = false;
    bool     teardown          = false;
    bool     aborted           = false;
    bool     frozen            = false;

    // Confirmacion en dos fases (begin_sync/end_sync)
    bool     sync_pending        = false;
    uint64_t sync_epoch_snapshot = 0;

    // Tracking de epochs y sincronización
    uint64_t fill_epoch = 0;
    uint64_t last_confirmed_epoch = 0;
    struct inflight_entry {
        llama_expert_key key;
        uint64_t epoch;
    };
    std::list<inflight_entry> l2_inflight;

    // Backend
    ggml_backend_t bound_backend = nullptr;
    enum class sync_mode { UNKNOWN, HOST_ONLY, CPU_SYNC, CALLBACK_SYNC };
    sync_mode effective_sync_mode = sync_mode::UNKNOWN;

    // Fail-Stop & Cuarentena
    std::atomic_flag fail_stop_claimed = ATOMIC_FLAG_INIT;
    std::atomic<bool> fatal{false};
    std::unique_ptr<quarantine_node> preallocated_quarantine;

    // Estadísticas
    llama_expert_stats stats;
    mutable llama_expert_arena_stats arena_stats;

    // Contadores de eventos
    uint64_t evictions         = 0;
    uint64_t evict_bytes       = 0;
    uint64_t evict_slack_bytes = 0;
    uint64_t sync_calls        = 0;
    uint64_t sync_failures     = 0;
    uint64_t alloc_fail        = 0;
    uint64_t oom_rejects       = 0;

    impl() = default;

    ~impl() {
        if (arena_base != nullptr) {
            drain_for_teardown();
            stats.log("shutdown");
        }

        for (auto & s : shards) {
            if (s.fd >= 0) {
                ::close(s.fd);
                s.fd = -1;
            }
        }

        if (arena_buf) {
            ggml_backend_buffer_free(arena_buf);
            arena_buf = nullptr;
            arena_base = nullptr;
        } else if (owns_arena && arena_base) {
            free(arena_base);
            arena_base = nullptr;
        }
    }

    [[noreturn]] void enter_fail_stop(fail_reason reason) noexcept {
        if (fail_stop_claimed.test_and_set(std::memory_order_acq_rel)) {
            for (;;) {
                std::this_thread::yield();
            }
        }

        fatal.store(true, std::memory_order_release);

        if (preallocated_quarantine) {
            quarantine_node * q = preallocated_quarantine.release();
            q->arena_buf   = arena_buf;
            q->arena_base  = arena_base;
            q->arena_bytes = arena_bytes;
            q->owns_arena  = owns_arena;
            q->backend     = bound_backend;
            q->reason      = reason;
            q->fail_stop   = true;

            g_arena_quarantine.enroll(q);
        }

        std::_Exit(EXIT_FAILURE);
    }

    // Retira la arena del uso sin liberarla. Se usa cuando no se puede demostrar
    // que el dispositivo ya no la esta leyendo (transferencia pendiente cuyo
    // resultado de sincronizacion no se puede observar, o barrera activa al
    // iniciar el teardown). La memoria se filtra a proposito: es preferible
    // perderla a devolverla al heap mientras Vulkan/CPU todavia la usa.
    void quarantine_arena(fail_reason reason) noexcept {
        arena_quarantined.store(true, std::memory_order_release);

        if (preallocated_quarantine) {
            quarantine_node * q = preallocated_quarantine.release();
            q->arena_buf   = arena_buf;
            q->arena_base  = arena_base;
            q->arena_bytes = arena_bytes;
            q->owns_arena  = owns_arena;
            q->backend     = bound_backend;
            q->reason      = reason;
            q->fail_stop   = false;

            g_arena_quarantine.enroll(q);
        }

        // pase lo que pase, el destructor no debe liberar esta memoria
        arena_buf  = nullptr;
        arena_base = nullptr;
        owns_arena = false;
    }


    void init_backend_memory(ggml_backend_t backend) {
        if (!backend || arena_buf != nullptr) {
            return;
        }
        // Solo se puede sustituir la reserva mientras no haya bloques vivos: los
        // punteros de la arena ya entregados dejarian de ser validos.
        if (!l2_index.empty()) {
            LLAMA_LOG_WARN("nexus-expert-stream: arena already in use, keeping PAGEABLE_HOST\n");
            return;
        }
#if defined(GGML_USE_VULKAN)
        if (ggml_backend_is_vk(backend)) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev) {
                ggml_backend_buffer_type_t expected_buft = ggml_backend_dev_host_buffer_type(dev);
                if (expected_buft != nullptr && expected_buft == ggml_backend_vk_host_buffer_type()) {
                    ggml_backend_buffer_t vk_buf = ggml_backend_buft_alloc_buffer(expected_buft, arena_bytes);
                    if (vk_buf != nullptr) {
                        ggml_backend_buffer_type_t effective_buft = ggml_backend_buffer_get_type(vk_buf);
                        void * new_base = ggml_backend_buffer_get_base(vk_buf);
                        if (effective_buft == expected_buft && new_base != nullptr) {
                            // Escenario 1: PINNED_VK exitoso
                            if (owns_arena && arena_base) {
                                free(arena_base);
                            }
                            arena_buf   = vk_buf;
                            arena_base  = new_base;
                            owns_arena  = false;
                            mem_kind    = llama_expert_mem_kind::PINNED_VK;
                            LLAMA_LOG_INFO("nexus-expert-stream: upgraded to PINNED_VK host buffer (%zu bytes)\n", arena_bytes);
                            return;
                        } else if (effective_buft == ggml_backend_cpu_buffer_type() && new_base != nullptr) {
                            // Escenario 2: PAGEABLE_HOST_FALLBACK devuelto por Vulkan
                            if (owns_arena && arena_base) {
                                free(arena_base);
                            }
                            arena_buf   = vk_buf;
                            arena_base  = new_base;
                            owns_arena  = false;
                            mem_kind    = llama_expert_mem_kind::PAGEABLE_HOST_FALLBACK;
                            LLAMA_LOG_WARN("nexus-expert-stream: Vulkan returned CPU fallback buffer (PAGEABLE_HOST_FALLBACK)\n");
                            return;
                        } else {
                            // Buffer devuelto no soportado / desconocido -> rechazar y liberar
                            LLAMA_LOG_ERROR("nexus-expert-stream: unsupported buffer type returned by Vulkan, rejecting\n");
                            ggml_backend_buffer_free(vk_buf);
                        }
                    }
                }
            }
        }
#endif
        // Sin backend Vulkan o sin host buft disponible: se mantiene la reserva
        // posix_memalign (PAGEABLE_HOST). No se afirma que sea memoria pinned.
        LLAMA_LOG_INFO("nexus-expert-stream: keeping PAGEABLE_HOST arena (no pinned host buffer available)\n");
    }

    // Allocator First-Fit
    bool alloc_block(size_t req_bytes, size_t & out_offset, size_t & out_size, size_t & out_slack) {
        size_t req = 0;
        if (!safe_align_up(req_bytes, alignment, req)) {
            return false;
        }

        for (auto it = free_holes.begin(); it != free_holes.end(); ++it) {
            size_t h_offset = it->first;
            size_t h_size   = it->second;

            if (h_size >= req) {
                size_t rem = h_size - req;
                out_offset = h_offset;
                out_size   = req;

                if (rem == 0) {
                    out_slack = 0;
                    free_holes.erase(it);
                } else if (rem < alignment) {
                    out_slack = rem;
                    free_holes.erase(it);
                } else {
                    out_slack = 0;
                    free_holes.erase(it);
                    free_holes[h_offset + req] = rem;
                }

                aligned_bytes += out_size;
                slack_bytes   += out_slack;
                free_bytes    -= (out_size + out_slack);
                return true;
            }
        }
        return false;
    }

    void free_block(size_t offset, size_t size, size_t slack) {
        size_t total = size + slack;
        aligned_bytes -= size;
        slack_bytes   -= slack;
        free_bytes    += total;

        auto next = free_holes.lower_bound(offset);
        auto prev = (next != free_holes.begin()) ? std::prev(next) : free_holes.end();

        size_t new_offset = offset;
        size_t new_size   = total;

        if (prev != free_holes.end() && prev->first + prev->second == offset) {
            new_offset = prev->first;
            new_size  += prev->second;
            free_holes.erase(prev);
        }

        if (next != free_holes.end() && offset + total == next->first) {
            new_size += next->second;
            free_holes.erase(next);
        }

        free_holes[new_offset] = new_size;
    }

    bool evict_lru_victim() {
        for (auto it = l2_lru_list.rbegin(); it != l2_lru_list.rend(); ++it) {
            auto idx_it = l2_index.find(*it);
            if (idx_it == l2_index.end()) continue;

            nexus_l2_block & blk = idx_it->second;
            if (blk.content == llama_expert_content::RESIDENT &&
                blk.pin_layers == 0 &&
                blk.inflight == 0 &&
                !blk.needs_proof &&
                last_confirmed_epoch >= blk.last_fill_epoch) {

                free_block(blk.offset, blk.size, blk.slack);
                evictions++;
                evict_bytes += blk.size;
                evict_slack_bytes += blk.slack;

                l2_lru_list.erase(blk.lru_it);
                l2_index.erase(idx_it);
                return true;
            }
        }
        return false;
    }

    bool resolve_sync(uint64_t /* sync_epoch */) {
        if (effective_sync_mode == sync_mode::UNKNOWN) {
            // Ningun fill enlazo un backend: no hay transferencia que confirmar.
            // Devolver false aqui marcaria un fallo de sincronizacion inexistente
            // al destruir una fuente que nunca se uso.
            return l2_inflight.empty();
        }
        if (effective_sync_mode == sync_mode::HOST_ONLY) {
            return l2_inflight.empty();
        }
        if (effective_sync_mode == sync_mode::CPU_SYNC) {
            if (bound_backend) {
                ggml_backend_synchronize(bound_backend);
            }
            return true;
        }
        if (effective_sync_mode == sync_mode::CALLBACK_SYNC) {
            try {
                ggml_backend_synchronize(bound_backend);
                return true;
            } catch (...) {
                return false;
            }
        }
        return false;
    }

    void process_inflight_upto(uint64_t sync_epoch) {
        for (auto it = l2_inflight.begin(); it != l2_inflight.end(); ) {
            if (it->epoch > sync_epoch) {
                ++it;
                continue;
            }
            auto b_it = l2_index.find(it->key);
            if (b_it == l2_index.end()) {
                enter_fail_stop(fail_reason::inflight_without_block);
            }
            nexus_l2_block & blk = b_it->second;
            if (blk.inflight == 0) {
                enter_fail_stop(fail_reason::inflight_underflow);
            }
            --blk.inflight;
            it = l2_inflight.erase(it);
        }
        last_confirmed_epoch = sync_epoch;
    }

    void synchronize_active_fill() {
        std::unique_lock<std::mutex> lk(l2_mutex);

        // El llamante NO cuenta en programming_fills: solo espera la compuerta.
        while (gate_closed) {
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
            const uint64_t my_epoch = gate_epoch;
            cv_gate.wait(lk, [&]{
                return teardown || aborted || fatal.load(std::memory_order_relaxed) || gate_epoch != my_epoch;
            });
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
        }

        gate_closed    = true;
        barrier_active = true;

        cv_quiescent.wait(lk, [&]{
            return teardown || aborted || fatal.load(std::memory_order_relaxed) || programming_fills == 0;
        });

        if (teardown || aborted || fatal.load(std::memory_order_relaxed)) {
            if (teardown) {
                gate_closed = true;
                barrier_active = false;
                return;
            }
            gate_closed = false;
            barrier_active = false;
            ++gate_epoch;
            cv_gate.notify_all();
            return;
        }

        // Con la compuerta cerrada y programming_fills == 0, l2_inflight es el
        // conjunto completo de transferencias no confirmadas. Si esta vacio no
        // hay nada que sincronizar: no se contabiliza una sincronizacion.
        if (l2_inflight.empty()) {
            barrier_active = false;
            gate_closed    = false;
            ++gate_epoch;
            cv_gate.notify_all();
            return;
        }

        ++sync_calls;
        const uint64_t sync_epoch = fill_epoch;
        lk.unlock();

        const bool sync_ok = resolve_sync(sync_epoch);

        lk.lock();
        if (!sync_ok) {
            ++sync_failures;
            lk.unlock();
            enter_fail_stop(fail_reason::sync_failed);
        }
        if (sync_epoch < last_confirmed_epoch) {
            // La contabilidad de epochs no puede retroceder: seria imposible
            // decidir que transferencias confirma esta sincronizacion.
            lk.unlock();
            enter_fail_stop(fail_reason::epoch_regression);
        }
        process_inflight_upto(sync_epoch);

        if (teardown) {
            gate_closed = true;
            barrier_active = false;
            return;
        }
        barrier_active = false;
        gate_closed    = false;
        ++gate_epoch;
        cv_gate.notify_all();
    }

    void synchronize_external() {
        std::unique_lock<std::mutex> lk(l2_mutex);

        while (gate_closed) {
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
            const uint64_t my_epoch = gate_epoch;
            cv_gate.wait(lk, [&]{
                return teardown || aborted || fatal.load(std::memory_order_relaxed) || gate_epoch != my_epoch;
            });
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
        }

        gate_closed    = true;
        barrier_active = true;

        cv_quiescent.wait(lk, [&]{
            return teardown || aborted || fatal.load(std::memory_order_relaxed) || programming_fills == 0;
        });

        if (teardown || aborted || fatal.load(std::memory_order_relaxed)) {
            if (teardown) {
                gate_closed = true;
                barrier_active = false;
                return;
            }
            gate_closed = false;
            barrier_active = false;
            ++gate_epoch;
            cv_gate.notify_all();
            return;
        }

        // Con la compuerta cerrada y programming_fills == 0, l2_inflight es el
        // conjunto completo de transferencias no confirmadas. Si esta vacio no
        // hay nada que sincronizar: no se contabiliza una sincronizacion.
        if (l2_inflight.empty()) {
            barrier_active = false;
            gate_closed    = false;
            ++gate_epoch;
            cv_gate.notify_all();
            return;
        }

        ++sync_calls;
        const uint64_t sync_epoch = fill_epoch;
        lk.unlock();

        const bool sync_ok = resolve_sync(sync_epoch);

        lk.lock();
        if (!sync_ok) {
            ++sync_failures;
            lk.unlock();
            enter_fail_stop(fail_reason::sync_failed);
        }
        if (sync_epoch < last_confirmed_epoch) {
            // La contabilidad de epochs no puede retroceder: seria imposible
            // decidir que transferencias confirma esta sincronizacion.
            lk.unlock();
            enter_fail_stop(fail_reason::epoch_regression);
        }
        process_inflight_upto(sync_epoch);

        if (teardown) {
            gate_closed = true;
            barrier_active = false;
            return;
        }
        barrier_active = false;
        gate_closed    = false;
        ++gate_epoch;
        cv_gate.notify_all();
    }

    // Cierra la compuerta y toma una instantanea del epoch. El llamante debe
    // ejecutar ggml_backend_synchronize() en el backend enlazado y luego end_sync().
    void begin_sync() {
        std::unique_lock<std::mutex> lk(l2_mutex);

        if (teardown || aborted || fatal.load(std::memory_order_relaxed)) {
            return;
        }
        if (sync_pending) {
            // begin_sync() no es reentrante: cada begin_sync() debe emparejarse
            // con su end_sync() antes de volver a llamarlo.
            LLAMA_LOG_ERROR("nexus-expert-stream: begin_sync() with a synchronization already pending\n");
            return;
        }
        // Sin backend enlazado no hay transferencias que confirmar, y antes del
        // primer fill tampoco (UNKNOWN).
        if (effective_sync_mode == sync_mode::HOST_ONLY ||
            effective_sync_mode == sync_mode::UNKNOWN) {
            return;
        }

        while (gate_closed) {
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
            const uint64_t my_epoch = gate_epoch;
            cv_gate.wait(lk, [&]{
                return teardown || aborted || fatal.load(std::memory_order_relaxed) || gate_epoch != my_epoch;
            });
            if (teardown || aborted || fatal.load(std::memory_order_relaxed)) return;
        }

        gate_closed    = true;
        barrier_active = true;

        cv_quiescent.wait(lk, [&]{
            return teardown || aborted || fatal.load(std::memory_order_relaxed) || programming_fills == 0;
        });

        if (teardown || aborted || fatal.load(std::memory_order_relaxed)) {
            gate_closed    = false;
            barrier_active = false;
            ++gate_epoch;
            cv_gate.notify_all();
            return;
        }

        sync_epoch_snapshot = fill_epoch;
        sync_pending        = true;
    }

    // El llamante ya sincronizo el backend enlazado.
    void end_sync() {
        std::unique_lock<std::mutex> lk(l2_mutex);

        if (!sync_pending) {
            return;
        }
        sync_pending = false;

        if (sync_epoch_snapshot < last_confirmed_epoch) {
            // La contabilidad de epochs no puede retroceder: seria imposible
            // decidir que transferencias confirma esta sincronizacion.
            lk.unlock();
            enter_fail_stop(fail_reason::epoch_regression);
        }

        ++sync_calls;
        process_inflight_upto(sync_epoch_snapshot);

        barrier_active = false;
        gate_closed    = false;
        ++gate_epoch;
        cv_gate.notify_all();
    }

    void drain_for_teardown() {
        std::unique_lock<std::mutex> lk(l2_mutex);

        teardown    = true;
        frozen      = true;
        gate_closed = true;
        sync_pending = false;

        cv_gate.notify_all();
        cv_quiescent.notify_all();

        cv_idle.wait(lk, [&]{ return active_fills == 0; });

        if (barrier_active) {
            // Otra hebra esta a mitad de una barrera (begin_sync() sin end_sync()).
            // Puede haber programado una transferencia y estar fuera del mutex
            // esperando al backend: cerrar la compuerta NO completa esa
            // transferencia, y ggml_backend_synchronize() no informa su resultado.
            // Politica explicita: la arena se retira sin liberarse (cuarentena)
            // en vez de devolver memoria que el dispositivo puede estar leyendo.
            // Destruir el objeto mientras otra hebra lo usa viola el contrato de
            // vida util del llamante; el destructor no puede repararlo.
            LLAMA_LOG_ERROR("nexus-expert-stream: teardown with an active barrier: "
                            "the L2 arena is quarantined instead of freed\n");
            quarantine_arena(fail_reason::lifetime_violation);
            return;
        }

        // active_fills == 0 y l2_inflight solo cambia bajo l2_mutex: si esta
        // vacio no queda ninguna transferencia programada sin confirmar.
        if (l2_inflight.empty()) {
            return;
        }

        ++sync_calls;
        const uint64_t sync_epoch = fill_epoch;
        lk.unlock();

        const bool sync_ok = resolve_sync(sync_epoch);

        lk.lock();
        if (!sync_ok) {
            // No se puede demostrar que las transferencias pendientes terminaron.
            // No se libera una arena que el dispositivo podria estar usando: se
            // deja en cuarentena.
            ++sync_failures;
            fatal.store(true, std::memory_order_release);
            LLAMA_LOG_ERROR("nexus-expert-stream: final synchronize failed during teardown: "
                            "the L2 arena is quarantined instead of freed\n");
            quarantine_arena(fail_reason::teardown_sync_failed);
            return;
        }
        process_inflight_upto(sync_epoch);
    }
};

struct llama_expert_source::impl::fill_active_guard {
    llama_expert_source::impl * im = nullptr;
    bool active_armed      = false;
    bool programming_armed = false;

    explicit fill_active_guard(llama_expert_source::impl * p) : im(p) {}
    void arm_active()      noexcept { active_armed = true; }
    void arm_programming() noexcept { programming_armed = true; }

    ~fill_active_guard() {
        if (active_armed && im) {
            std::lock_guard<std::mutex> lk(im->l2_mutex);
            --im->active_fills;
            if (programming_armed) {
                --im->programming_fills;
                programming_armed = false;
            }
            if (im->active_fills == 0) {
                im->cv_idle.notify_all();
            }
            im->cv_quiescent.notify_all();
        }
    }
};

std::unique_ptr<llama_expert_source> llama_expert_source::create(
        const llama_expert_model_meta & meta,
        const config & cfg) {

    if (::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr) {
        LLAMA_LOG_ERROR("nexus: GGML_VK_ASYNC_USE_TRANSFER_QUEUE must be unset for D-0; dedicated transfer queue excluded\n");
        return nullptr;
    }
    if (cfg.path_2) {
        LLAMA_LOG_ERROR("nexus: config.path_2 is excluded until X-02 is certified\n");
        return nullptr;
    }

    std::unique_ptr<quarantine_node> q_node;
    try {
        quarantine_node * q_raw = new (std::nothrow) quarantine_node();
        if (!q_raw) {
            LLAMA_LOG_ERROR("nexus-expert-stream: failed to preallocate quarantine node\n");
            return nullptr;
        }
        q_node.reset(q_raw);
    } catch (...) {
        return nullptr;
    }

    std::unique_ptr<impl> p;
    try {
        p = std::make_unique<impl>();
        p->preallocated_quarantine = std::move(q_node);
    } catch (...) {
        return nullptr;
    }

    p->cfg = cfg;

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

    p->extents = meta.tensors;
    for (const auto & kv : p->extents) {
        if (!kv.second.is_supported) {
            continue;
        }
        size_t sz = (size_t) kv.second.expert_bytes;
        p->max_expert_size = std::max(p->max_expert_size, sz);
        if (sz >= 2000000) {
            p->size_gate_up = sz;
        } else {
            p->size_down = sz;
        }
    }

    if (p->max_expert_size == 0) {
        LLAMA_LOG_ERROR("nexus-expert-stream: no supported MoE expert tensors found\n");
        return nullptr;
    }

    if (p->size_gate_up == 0) p->size_gate_up = p->max_expert_size;
    if (p->size_down == 0)    p->size_down    = p->max_expert_size;

    p->arena_bytes = cfg.ram_warm_bytes;
    p->alignment   = cfg.alignment > 0 ? cfg.alignment : 64;

    // Inicializar reserva host independiente con posix_memalign
    constexpr size_t MAX_ALIGN = ((size_t) 1) << (std::numeric_limits<size_t>::digits - 1);
    if (p->alignment == 0 || (p->alignment & (p->alignment - 1)) != 0 ||
        p->alignment % sizeof(void *) != 0 || p->alignment > MAX_ALIGN) {
        LLAMA_LOG_ERROR("nexus-expert-stream: invalid alignment requested: %zu\n", p->alignment);
        return nullptr;
    }

    int res = posix_memalign(&p->arena_base, p->alignment, p->arena_bytes);
    if (res != 0 || !p->arena_base) {
        LLAMA_LOG_ERROR("nexus-expert-stream: failed to allocate L2 arena (%zu bytes): %s\n",
            p->arena_bytes, strerror(res));
        return nullptr;
    }

    p->owns_arena = true;
    p->arena_buf  = nullptr;
    p->mem_kind   = llama_expert_mem_kind::PAGEABLE_HOST;

    p->free_holes[0] = p->arena_bytes;
    p->aligned_bytes = 0;
    p->slack_bytes   = 0;
    p->free_bytes    = p->arena_bytes;

    LLAMA_LOG_INFO(
        "nexus-expert-stream: L2 single arena initialized: %.2f MiB (alignment: %zu, first-fit coalescing)\n",
        (double) p->arena_bytes / (1024.0 * 1024.0), p->alignment);

    auto result = std::make_unique<llama_expert_source>();
    result->pimpl = std::move(p);
    return result;
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
    impl::fill_active_guard guard(&p);

    {
        std::lock_guard<std::mutex> lk(p.l2_mutex);
        if (p.teardown || p.frozen || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
            return llama_expert_fill_status::IO_ERROR;
        }
        ++p.active_fills;
        guard.arm_active();

        // B3: Comprobación estricta de backend
        if (p.bound_backend != nullptr && backend != nullptr && p.bound_backend != backend) {
            LLAMA_LOG_ERROR("nexus-expert-stream: backend mismatch: operation requested on a different backend\n");
            return llama_expert_fill_status::IO_ERROR;
        }

        // Enlazar backend y clasificar modo en el primer fill
        if (!p.bound_backend && backend) {
            p.bound_backend = backend;
#if defined(GGML_USE_VULKAN)
            if (ggml_backend_is_vk(backend)) {
                p.effective_sync_mode = impl::sync_mode::CALLBACK_SYNC;
                p.init_backend_memory(backend);
            } else
#endif
            {
                p.effective_sync_mode = impl::sync_mode::CPU_SYNC;
            }
        } else if (!p.bound_backend && !backend) {
            p.effective_sync_mode = impl::sync_mode::HOST_ONLY;
        }
    }

    auto it_ext = p.extents.find(w);
    if (it_ext == p.extents.end() || !it_ext->second.is_supported) {
        return llama_expert_fill_status::FALLBACK_ALLOWED;
    }

    if (expert_size > p.arena_bytes) {
        return llama_expert_fill_status::NOT_SUPPORTED;
    }

    p.stats.fills++;
    llama_expert_key key { il, reinterpret_cast<uintptr_t>(w), expert };

    // --- 1. Consulta L2 Cache & Coordinación LOADING (B2) ---
    bool is_hit = false;
    size_t blk_offset = 0;
    {
        std::unique_lock<std::mutex> lk(p.l2_mutex);
        while (true) {
            auto it = p.l2_index.find(key);
            if (it != p.l2_index.end()) {
                if (it->second.content == llama_expert_content::RESIDENT) {
                    is_hit = true;
                    p.stats.l2_hits++;
                    nexus_l2_block & blk = it->second;
                    blk.pin_layers++;
                    blk.lru_tick = ++p.current_lru_tick;
                    blk_offset = blk.offset;

                    p.l2_lru_list.erase(blk.lru_it);
                    p.l2_lru_list.push_front(key);
                    blk.lru_it = p.l2_lru_list.begin();
                    break;
                } else if (it->second.content == llama_expert_content::LOADING) {
                    // B2: Otro hilo está cargando esta clave. Esperar en cv_loading.
                    p.cv_loading.wait(lk);
                    if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
                        return llama_expert_fill_status::IO_ERROR;
                    }
                    continue; // Re-evaluar estado tras el despertar
                } else {
                    // INVALID o estado anómalo
                    return llama_expert_fill_status::IO_ERROR;
                }
            } else {
                // Clave ausente en L2 -> requiere carga
                break;
            }
        }

    }

    // --- 2. Miss: Asignación y carga NVMe ---
    if (!is_hit) {
        p.stats.l2_misses++;
        size_t alloc_off = 0;
        size_t alloc_sz  = 0;
        size_t alloc_slk = 0;

        {
            std::unique_lock<std::mutex> ul(p.l2_mutex);
            while (!p.alloc_block(expert_size, alloc_off, alloc_sz, alloc_slk)) {
                if (!p.evict_lru_victim()) {
                    ul.unlock();
                    p.synchronize_active_fill();
                    ul.lock();

                    if (!p.evict_lru_victim()) {
                        p.alloc_fail++;
                        p.oom_rejects++;
                        return llama_expert_fill_status::OUT_OF_MEMORY;
                    }
                }
            }

            nexus_l2_block blk;
            blk.offset     = alloc_off;
            blk.size       = alloc_sz;
            blk.slack      = alloc_slk;
            blk.req_bytes  = expert_size;
            blk.content    = llama_expert_content::LOADING;
            blk.pin_layers = 1;
            blk.inflight   = 0;
            blk.lru_tick   = ++p.current_lru_tick;

            p.l2_lru_list.push_front(key);
            blk.lru_it = p.l2_lru_list.begin();
            p.l2_index[key] = blk;
            blk_offset = alloc_off;
        }

        // Lectura desde NVMe fuera de l2_mutex
        const auto & ext = it_ext->second;
        const int fd = p.shards[ext.shard_idx].fd;
        const uint64_t file_offset = ext.base_offset + (uint64_t) expert * ext.expert_stride;
        const uint64_t file_size   = p.shards[ext.shard_idx].size;

        void * dst_ptr = static_cast<uint8_t *>(p.arena_base) + blk_offset;

        const auto t0 = std::chrono::steady_clock::now();
        int64_t n_read = llama_expert_pread_full(fd, dst_ptr, expert_size, file_offset, file_size);
        const auto t1 = std::chrono::steady_clock::now();

        if (n_read != (int64_t) expert_size) {
            LLAMA_LOG_ERROR("nexus-expert-stream: pread_full failed for %s exp %d\n", ggml_get_name(w), expert);
            std::lock_guard<std::mutex> lk(p.l2_mutex);
            auto it = p.l2_index.find(key);
            if (it != p.l2_index.end()) {
                p.free_block(it->second.offset, it->second.size, it->second.slack);
                p.l2_lru_list.erase(it->second.lru_it);
                p.l2_index.erase(it);
            }
            p.cv_loading.notify_all(); // B2: Despertar cualquier hilo que esperaba
            return llama_expert_fill_status::IO_ERROR;
        }

        const uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        p.stats.proc_bytes += expert_size;
        p.stats.io_us_total += us;

        uint64_t cur_max = p.stats.io_us_max.load();
        while (us > cur_max && !p.stats.io_us_max.compare_exchange_weak(cur_max, us)) {
        }

        // Transición atómica de LOADING a RESIDENT
        {
            std::lock_guard<std::mutex> lk(p.l2_mutex);
            auto it = p.l2_index.find(key);
            if (it != p.l2_index.end()) {
                it->second.content = llama_expert_content::RESIDENT;
            }
            p.cv_loading.notify_all(); // B2: Notificar a hilos esperando por esta clave
        }
    }

    // --- 3. Programación de la transferencia VRAM con Protocolo de Barrera ---
    if (backend && bank) {
        std::unique_lock<std::mutex> lk(p.l2_mutex);

        if (p.fatal.load(std::memory_order_relaxed)) {
            return llama_expert_fill_status::IO_ERROR;
        }

        // Esta hebra aun NO programa nada: no cuenta en programming_fills, solo
        // espera a que la barrera en curso abra la compuerta.
        while (p.gate_closed) {
            if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
                return llama_expert_fill_status::IO_ERROR;
            }
            const uint64_t my_epoch = p.gate_epoch;
            p.cv_gate.wait(lk, [&]{
                return p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed) || p.gate_epoch != my_epoch;
            });
            if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
                return llama_expert_fill_status::IO_ERROR;
            }
        }

        if (p.fatal.load(std::memory_order_relaxed)) {
            return llama_expert_fill_status::IO_ERROR;
        }

        // Protección defensiva contra overflow de epochs
        if (p.fill_epoch == UINT64_MAX) {
            p.enter_fail_stop(fail_reason::epoch_overflow);
        }

        // A partir de aquí la hebra programa una transferencia: cuenta para la
        // quiescencia de las barreras hasta que regrese de set_async.
        ++p.programming_fills;
        guard.arm_programming();

        auto it = p.l2_index.find(key);
        if (it != p.l2_index.end()) {
            uint64_t next_epoch = p.fill_epoch + 1;
            try {
                p.l2_inflight.push_back({ key, next_epoch });
            } catch (...) {
                // Si la asignación de memoria para l2_inflight falla, no alterar contadores
                return llama_expert_fill_status::OUT_OF_MEMORY;
            }
            p.fill_epoch = next_epoch;
            it->second.inflight++;
            it->second.last_fill_epoch = next_epoch;
        }

        lk.unlock();

        void * src_ptr = static_cast<uint8_t *>(p.arena_base) + blk_offset;
        ggml_backend_tensor_set_async(backend, bank, src_ptr, slot_offset, expert_size);
    }

    return llama_expert_fill_status::LOADED;
}

void llama_expert_source::on_layer_synchronized(int32_t il) {
    auto & p = *pimpl;
    std::lock_guard<std::mutex> lk(p.l2_mutex);
    for (auto & kv : p.l2_index) {
        if (kv.second.pin_layers > 0 && kv.first.layer_id <= il) {
            kv.second.pin_layers = 0;
        }
    }
}

void llama_expert_source::synchronize_all() {
    pimpl->synchronize_external();
}

void llama_expert_source::drain() {
    pimpl->drain_for_teardown();
}

bool llama_expert_source::was_arena_quarantined() const {
    return pimpl->arena_quarantined.load(std::memory_order_acquire);
}

void llama_expert_source::begin_sync() {
    pimpl->begin_sync();
}

void llama_expert_source::end_sync() {
    pimpl->end_sync();
}

const llama_expert_stats & llama_expert_source::get_stats() const {
    return pimpl->stats;
}

const llama_expert_arena_stats & llama_expert_source::get_arena_stats() const {
    auto & p = *pimpl;
    std::lock_guard<std::mutex> lk(p.l2_mutex);

    p.arena_stats.arena_bytes.store(p.arena_bytes);
    p.arena_stats.aligned_bytes.store(p.aligned_bytes);
    p.arena_stats.slack_bytes.store(p.slack_bytes);
    p.arena_stats.free_bytes.store(p.free_bytes);

    size_t resident = 0, invalid = 0, pinned = 0, inflight = 0;
    size_t bytes_l = 0;
    for (const auto & kv : p.l2_index) {
        const auto & b = kv.second;
        if (b.content == llama_expert_content::RESIDENT) resident++;
        if (b.content == llama_expert_content::LOADING)  { bytes_l += b.size; }
        if (b.content == llama_expert_content::INVALID)  invalid++;
        if (b.pin_layers > 0) pinned++;
        if (b.inflight > 0)   inflight++;
    }

    p.arena_stats.blocks_live.store(p.l2_index.size());
    p.arena_stats.blocks_resident.store(resident);
    p.arena_stats.blocks_invalid.store(invalid);
    p.arena_stats.blocks_pinned.store(pinned);
    p.arena_stats.blocks_inflight.store(inflight);
    p.arena_stats.bytes_loading.store(bytes_l);

    size_t largest_hole = 0;
    for (const auto & h : p.free_holes) {
        largest_hole = std::max(largest_hole, h.second);
    }
    p.arena_stats.largest_free_hole.store(largest_hole);
    p.arena_stats.align_a.store(p.alignment);

    p.arena_stats.evictions.store(p.evictions);
    p.arena_stats.evict_bytes.store(p.evict_bytes);
    p.arena_stats.evict_slack_bytes.store(p.evict_slack_bytes);
    p.arena_stats.sync_calls.store(p.sync_calls);
    p.arena_stats.sync_failures.store(p.sync_failures);
    p.arena_stats.alloc_fail.store(p.alloc_fail);
    p.arena_stats.oom_rejects.store(p.oom_rejects);

    return p.arena_stats;
}

bool llama_expert_source::get_block_info(const llama_expert_key & key, llama_expert_block_info & out) const {
    auto & p = *pimpl;
    std::lock_guard<std::mutex> lk(p.l2_mutex);
    auto it = p.l2_index.find(key);
    if (it == p.l2_index.end()) {
        return false;
    }
    const auto & b = it->second;
    out.content     = b.content;
    out.pin_layers  = b.pin_layers;
    out.inflight    = b.inflight;
    out.needs_proof = b.needs_proof;
    out.lru_tick    = b.lru_tick;
    out.offset      = b.offset;
    out.size        = b.size;
    out.slack       = b.slack;
    out.mem_kind    = p.mem_kind;
    return true;
}

size_t llama_expert_source::get_l2_capacity_pairs() const {
    auto & p = *pimpl;
    const size_t pair_sz = p.size_gate_up + p.size_down;
    if (pair_sz == 0) return 0;
    return p.arena_bytes / pair_sz;
}

size_t llama_expert_source::get_l2_block_count() const {
    std::lock_guard<std::mutex> lk(pimpl->l2_mutex);
    return pimpl->l2_index.size();
}

size_t llama_expert_source::get_arena_bytes() const {
    return pimpl->arena_bytes;
}

llama_expert_mem_kind llama_expert_source::get_l2_mem_kind() const {
    return pimpl->mem_kind;
}
