#include "llama-expert-source.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
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

void llama_expert_stats::record(llama_expert_fill_status st) {
    switch (st) {
        case llama_expert_fill_status::LOADED:
            // ya contabilizado como fill en el camino de exito
            break;
        case llama_expert_fill_status::FALLBACK_ALLOWED:
            fallback_mmap.fetch_add(1, std::memory_order_relaxed);
            break;
        case llama_expert_fill_status::NOT_SUPPORTED:
            not_supported.fetch_add(1, std::memory_order_relaxed);
            break;
        case llama_expert_fill_status::OUT_OF_MEMORY:
            out_of_memory.fetch_add(1, std::memory_order_relaxed);
            break;
        case llama_expert_fill_status::IO_ERROR:
            rejected_io.fetch_add(1, std::memory_order_relaxed);
            break;
        case llama_expert_fill_status::INVALID_EXTENT:
            rejected_invalid.fetch_add(1, std::memory_order_relaxed);
            break;
    }
}

void llama_expert_stats::log(const char * tag) const {
    const uint64_t attempts  = fill_attempts.load();
    const uint64_t total     = fills.load();
    const uint64_t reads     = completed_reads.load();
    const uint64_t programmed = uploads_programmed.load();
    const uint64_t confirmed = uploads_confirmed.load();
    const uint64_t l2_h      = l2_hits.load();
    const uint64_t l2_m      = l2_misses.load();
    const uint64_t bytes     = proc_bytes.load();
    const uint64_t uploaded  = upload_bytes.load();
    const uint64_t us        = io_us_total.load();
    const uint64_t max_u     = io_us_max.load();
    const uint64_t fallback  = fallback_mmap.load();
    const uint64_t not_sup   = not_supported.load();
    const uint64_t oom       = out_of_memory.load();
    const uint64_t rej_io    = rejected_io.load();
    const uint64_t rej_inv   = rejected_invalid.load();

    if (attempts == 0 && total == 0 && l2_h == 0 && l2_m == 0 && fallback == 0 && not_sup == 0 &&
        oom == 0 && rej_io == 0 && rej_inv == 0) {
        return;
    }

    // El hit rate de L2 se calcula sobre las consultas a la arena (hits + misses),
    // no sobre `fills`: los fills rechazados o en fallback no consultaron L2.
    // Los aciertos de L1 (banco del dispositivo) los reporta llama_moe_cache.
    const uint64_t lookups  = l2_h + l2_m;
    const double   hit_rate = lookups > 0 ? 100.0 * (double) l2_h / (double) lookups : 0.0;
    const double   avg_us   = reads > 0 ? (double) us / (double) reads : 0.0;
    const double   mb       = (double) bytes / (1024.0 * 1024.0);
    const double   up_mb    = (double) uploaded / (1024.0 * 1024.0);
    const double   thr      = us > 0 ? mb / ((double) us / 1000000.0) : 0.0;

    LLAMA_LOG_INFO(
        "nexus-expert-stream [%s]: attempts=%lu fills=%lu reads=%lu uploads=%lu confirmed=%lu | L2: hit=%lu miss=%lu (hit_rate=%.1f%%)\n",
        tag, (unsigned long) attempts, (unsigned long) total, (unsigned long) reads,
        (unsigned long) programmed, (unsigned long) confirmed,
        (unsigned long) l2_h, (unsigned long) l2_m, hit_rate);
    LLAMA_LOG_INFO(
        "nexus-expert-stream [%s]: NVMe: %.2f MiB in %lu completed reads, avg=%.1f us, max=%lu us (%.2f MiB/s) | uploaded: %.2f MiB\n",
        tag, mb, (unsigned long) reads, avg_us, (unsigned long) max_u, thr, up_mb);
    LLAMA_LOG_INFO(
        "nexus-expert-stream [%s]: fallback mmap=%lu not_supported=%lu oom=%lu rejected_io=%lu rejected_invalid=%lu | drops=%lu verify ok=%lu fail=%lu\n",
        tag, (unsigned long) fallback, (unsigned long) not_sup, (unsigned long) oom,
        (unsigned long) rej_io, (unsigned long) rej_inv,
        (unsigned long) drops.load(), (unsigned long) verify_ok.load(), (unsigned long) verify_fail.load());
}

void llama_expert_stats::reset() {
    fill_attempts.store(0);
    fills.store(0);
    completed_reads.store(0);
    uploads_programmed.store(0);
    uploads_confirmed.store(0);
    l2_hits.store(0);
    l2_misses.store(0);
    proc_bytes.store(0);
    upload_bytes.store(0);
    io_us_total.store(0);
    io_us_max.store(0);
    fallback_mmap.store(0);
    not_supported.store(0);
    out_of_memory.store(0);
    rejected_io.store(0);
    rejected_invalid.store(0);
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

// Gancho de inyeccion de fallos TEST-ONLY. Cuando g_nexus_test_alloc_fail >= 0,
// cada fase asignadora que lo consulta consume una unidad; la fase que llega a
// 0 falla con OUT_OF_MEMORY (o devuelve false en el allocator). -1 (default)
// desactiva el gancho. No afecta al funcionamiento normal ni a los hilos
// productivos (solo las pruebas lo arman).
static std::atomic<long long> g_nexus_test_alloc_fail{-1};

static bool nexus_test_fail_alloc() noexcept {
    const long long v = g_nexus_test_alloc_fail.load(std::memory_order_relaxed);
    if (v == -2) {
        // modo "fallar siempre hasta desarmar": solo lo usan las pruebas
        return true;
    }
    if (v < 0) {
        return false;
    }
    return g_nexus_test_alloc_fail.fetch_sub(1, std::memory_order_relaxed) == 0;
}

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

    size_t max_expert_size = 0;
    // bytes de un par de expertos de una capa (gate_up, o gate+up, mas down);
    // 0 = capacidad desconocida porque los roles/capas no estan identificados
    size_t pair_bytes      = 0;

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
                    // Bloqueante C: la alineacion activa NO se modifica hasta que
                    // la reserva nueva este confirmada y comprometida. Cualquier
                    // fallo conserva la arena posix original, su alineacion y su
                    // clasificacion de memoria honesta.
                    ggml_backend_buffer_t vk_buf = ggml_backend_buft_alloc_buffer(expected_buft, arena_bytes);
                    if (vk_buf != nullptr) {
                        // Bloqueante D: durante esta ventana existen dos reservas
                        // completas (posix + Vulkan). Es el pico transitorio de la
                        // sustitucion transaccional: no se libera la arena activa
                        // antes de tener una nueva validada. Pico = 2 * arena_bytes.
                        ggml_backend_buffer_type_t effective_buft = ggml_backend_buffer_get_type(vk_buf);
                        const size_t effective_alignment = ggml_backend_buffer_get_alignment(vk_buf);
                        void * new_base = ggml_backend_buffer_get_base(vk_buf);
                        if (effective_alignment == 0 ||
                            (effective_alignment & (effective_alignment - 1)) != 0 ||
                            effective_alignment % sizeof(void *) != 0 ||
                            new_base == nullptr) {
                            LLAMA_LOG_ERROR("nexus-expert-stream: invalid allocated Vulkan buffer (alignment %zu, base %p), "
                                            "keeping the pageable arena\n", effective_alignment, (void *) new_base);
                            ggml_backend_buffer_free(vk_buf);
                            return;
                        }
                        if (effective_buft == expected_buft) {
                            // Escenario 1: PINNED_VK exitoso. Se adopta la alineacion
                            // del buffer nuevo y se libera la reserva anterior.
                            if (owns_arena && arena_base) {
                                free(arena_base);
                            }
                            arena_buf   = vk_buf;
                            arena_base  = new_base;
                            owns_arena  = false;
                            alignment   = effective_alignment;
                            mem_kind    = llama_expert_mem_kind::PINNED_VK;
                            LLAMA_LOG_INFO("nexus-expert-stream: upgraded to PINNED_VK host buffer (%zu bytes, alignment %zu)\n",
                                    arena_bytes, alignment);
                            return;
                        }
                        if (effective_buft == ggml_backend_cpu_buffer_type()) {
                            // Escenario 2: PAGEABLE_HOST_FALLBACK devuelto por Vulkan
                            if (owns_arena && arena_base) {
                                free(arena_base);
                            }
                            arena_buf   = vk_buf;
                            arena_base  = new_base;
                            owns_arena  = false;
                            alignment   = effective_alignment;
                            mem_kind    = llama_expert_mem_kind::PAGEABLE_HOST_FALLBACK;
                            LLAMA_LOG_WARN("nexus-expert-stream: Vulkan returned CPU fallback buffer (PAGEABLE_HOST_FALLBACK, alignment %zu)\n",
                                    alignment);
                            return;
                        }
                        // Buffer devuelto no soportado / desconocido -> rechazar y
                        // conservar la reserva y alineacion originales.
                        LLAMA_LOG_ERROR("nexus-expert-stream: unsupported buffer type returned by Vulkan, "
                                        "keeping the pageable arena\n");
                        ggml_backend_buffer_free(vk_buf);
                    } else {
                        LLAMA_LOG_ERROR("nexus-expert-stream: Vulkan host buffer allocation failed, "
                                        "keeping the pageable arena\n");
                    }
                }
            }
        }
#endif
        // Sin backend Vulkan o sin host buft disponible: se mantiene la reserva
        // posix_memalign (PAGEABLE_HOST). No se afirma que sea memoria pinned.
        LLAMA_LOG_INFO("nexus-expert-stream: keeping PAGEABLE_HOST arena (no pinned host buffer available)\n");
    }

    // Allocator First-Fit. Las mutaciones de free_holes usan extract()/insert()
    // con node_type, que no asignan: un bad_alloc nunca deja el allocator a
    // medio modificar.
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

                auto node = free_holes.extract(it); // noexcept
                if (rem == 0) {
                    out_slack = 0;
                } else if (rem < alignment) {
                    out_slack = rem;
                } else {
                    out_slack = 0;
                    node.key()    = h_offset + req;
                    node.mapped() = rem;
                    free_holes.insert(std::move(node)); // sin asignacion: noexcept
                }

                aligned_bytes += out_size;
                slack_bytes   += out_slack;
                free_bytes    -= (out_size + out_slack);
                return true;
            }
        }
        return false;
    }

    // Libera [offset, size+slack) de vuelta a free_holes. Devuelve true solo si
    // el hueco quedo reinsertado y los contadores actualizados. NUNCA deja el
    // allocator a medio modificar:
    //   - caso con vecinos: los nodos se extraen (noexcept) y el hueco fusionado
    //     se reinserta reutilizando el nodo extraido, sin ninguna asignacion;
    //   - caso de hueco nuevo: la unica operacion asignadora se intenta antes de
    //     tocar nada; si falla, no se modifica ni free_holes ni los contadores.
    bool free_block(size_t offset, size_t size, size_t slack) {
        size_t total = size + slack;

        auto next = free_holes.lower_bound(offset);
        auto prev = (next != free_holes.begin()) ? std::prev(next) : free_holes.end();

        const bool merges_prev = prev != free_holes.end() && prev->first + prev->second == offset;
        const bool merges_next = next != free_holes.end() && offset + total == next->first;

        if (merges_prev) {
            auto node = free_holes.extract(prev); // noexcept
            size_t new_offset = node.key();
            size_t new_size   = total + node.mapped();
            if (merges_next) {
                // `next` puede haberse invalidado al extraer `prev`; rebuscarlo.
                auto it2 = free_holes.lower_bound(offset);
                if (it2 != free_holes.end() && offset + total == it2->first) {
                    auto node2 = free_holes.extract(it2); // noexcept
                    new_size += node2.mapped();
                }
            }
            node.key()    = new_offset;
            node.mapped() = new_size;
            free_holes.insert(std::move(node)); // reutiliza el nodo: noexcept
        } else if (merges_next) {
            auto node = free_holes.extract(next); // noexcept
            node.key()    = offset;
            node.mapped() = total + node.mapped();
            free_holes.insert(std::move(node)); // noexcept
        } else {
            // Hueco completamente nuevo: unica operacion asignadora. Se intenta
            // antes de tocar contadores; un fallo no modifica nada.
            if (nexus_test_fail_alloc()) {
                // inyeccion TEST-ONLY: la insercion del hueco falla deterministamente
                return false;
            }
            try {
                free_holes.emplace(offset, total);
            } catch (...) {
                return false;
            }
        }

        aligned_bytes -= size;
        slack_bytes   -= slack;
        free_bytes    += total;
        return true;
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

                // Solo se despublica el bloque si la liberacion confirmo exito.
                // Si falla, el bloque, la lista LRU, los indices y los contadores
                // quedan integros (sin perdida silenciosa de capacidad).
                if (!free_block(blk.offset, blk.size, blk.slack)) {
                    return false;
                }
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
            if (!bound_backend || ggml_backend_dev_type(ggml_backend_get_device(bound_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                return false;
            }
            ggml_backend_synchronize(bound_backend);
            return true;
        }
        if (effective_sync_mode == sync_mode::CALLBACK_SYNC) {
#if defined(GGML_USE_VULKAN)
            if (!bound_backend || !ggml_backend_is_vk(bound_backend)) {
                return false;
            }
#else
            return false;
#endif
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
            stats.uploads_confirmed.fetch_add(1, std::memory_order_relaxed);
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

void llama_expert_source::test_arm_alloc_failure(long long phases) {
    g_nexus_test_alloc_fail.store(phases, std::memory_order_relaxed);
}

void llama_expert_source::test_disarm_alloc_failure() {
    g_nexus_test_alloc_fail.store(-1, std::memory_order_relaxed);
}

struct llama_expert_create_result llama_expert_source::create(
        const llama_expert_model_meta & meta,
        const config & cfg) {

    llama_expert_create_result result;

    if (::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr) {
        LLAMA_LOG_ERROR("nexus: GGML_VK_ASYNC_USE_TRANSFER_QUEUE must be unset for D-0; dedicated transfer queue excluded\n");
        result.status = llama_expert_create_status::IO_ERROR;
        return result;
    }
    if (cfg.path_2) {
        LLAMA_LOG_ERROR("nexus: config.path_2 is excluded until X-02 is certified\n");
        result.status = llama_expert_create_status::IO_ERROR;
        return result;
    }

    std::unique_ptr<quarantine_node> q_node;
    try {
        quarantine_node * q_raw = new (std::nothrow) quarantine_node();
        if (!q_raw) {
            LLAMA_LOG_ERROR("nexus-expert-stream: failed to preallocate quarantine node\n");
            result.status = llama_expert_create_status::OUT_OF_MEMORY;
            return result;
        }
        q_node.reset(q_raw);
    } catch (...) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }

    std::unique_ptr<impl> p;
    try {
        p = std::make_unique<impl>();
        p->preallocated_quarantine = std::move(q_node);
    } catch (...) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }

    try {

    p->cfg = cfg;

    // fase asignadora 0: vector de shards
    if (nexus_test_fail_alloc()) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }
    p->shards.resize(meta.files.size());
    for (size_t i = 0; i < meta.files.size(); ++i) {
        p->shards[i].path = meta.files[i].path;
        p->shards[i].size = meta.files[i].size;
        p->shards[i].fd   = ::open(p->shards[i].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (p->shards[i].fd < 0) {
            LLAMA_LOG_ERROR("nexus-expert-stream: cannot open shard %s: %s\n",
                p->shards[i].path.c_str(), strerror(errno));
            result.status = llama_expert_create_status::IO_ERROR;
            return result;
        }
    }

    // fase asignadora 1: copia del mapa de metadatos
    if (nexus_test_fail_alloc()) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }
    p->extents = meta.tensors;

    // B4: la identidad gate/up vs down viene de metadatos explicitos del tensor
    // (rol y capa), nunca de un umbral de tamano. Un "par" es el conjunto de
    // expertos que una capa necesita: gate_up, o gate + up, mas down.
    struct layer_expert_bytes {
        size_t total    = 0;
        bool   has_down = false;
        bool   has_up   = false;
    };
    std::map<int32_t, layer_expert_bytes> per_layer;
    bool capacity_known = true;

    // fase asignadora 2: estructuras auxiliares de clasificacion/per_layer
    if (nexus_test_fail_alloc()) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }

    // Bloqueante B: clasificar el modelo antes de decidir. Un candidato MoE con
    // metadatos INVALID no es "ausencia de soporte": es corrupcion y debe
    // propagarse como error, no como fallback mmap.
    size_t n_invalid = 0;
    size_t n_supported = 0;
    size_t n_unsupported = 0;

    for (const auto & kv : p->extents) {
        const auto & m = kv.second;
        switch (m.validation) {
            case llama_expert_tensor_validation::INVALID:
                ++n_invalid;
                break;
            case llama_expert_tensor_validation::VALID:
                ++n_supported;
                break;
            case llama_expert_tensor_validation::UNSUPPORTED:
                ++n_unsupported;
                break;
        }
        if (m.is_supported) {
            ++n_supported;
        }
        // Revalidar los limites que este fuente servira. Un shard fuera de rango
        // o un extent fuera del archivo no puede declararse "soportado": seria
        // un fallback silencioso sobre metadatos corruptos.
        if (m.is_supported) {
            if (m.shard_idx < 0 || (size_t) m.shard_idx >= p->shards.size()) {
                ++n_invalid;
                continue;
            }
            const uint64_t shard_sz = p->shards[m.shard_idx].size;
            if (m.expert_bytes == 0 || m.expert_stride < m.expert_bytes ||
                m.n_expert <= 0 ||
                (m.expert_stride != 0 &&
                 (uint64_t) m.n_expert > (UINT64_MAX - m.base_offset) / m.expert_stride) ||
                m.base_offset > shard_sz ||
                (uint64_t) m.expert_bytes > shard_sz - m.base_offset) {
                ++n_invalid;
            }
        }
    }

    if (n_invalid > 0) {
        LLAMA_LOG_ERROR("nexus-expert-stream: %zu MoE tensor(s) with invalid metadata; "
                        "refusing to fall back to mmap\n", n_invalid);
        result.status = llama_expert_create_status::INVALID_METADATA;
        return result;
    }
    if (n_supported == 0) {
        LLAMA_LOG_INFO("nexus-expert-stream: no supported MoE expert tensors (%zu unsupported); "
                       "Nexus disabled, mmap fallback allowed\n", n_unsupported);
        result.status = llama_expert_create_status::NO_SUPPORTED_TENSORS;
        return result;
    }

    for (const auto & kv : p->extents) {
        const auto & m = kv.second;
        if (!m.is_supported) {
            continue;
        }
        p->max_expert_size = std::max(p->max_expert_size, (size_t) m.expert_bytes);

        if (m.layer_id < 0 || m.role == llama_expert_tensor_role::UNKNOWN) {
            // sin rol/capa explicitos no se sabe que tensores forman un par
            capacity_known = false;
            continue;
        }
        auto & lb = per_layer[m.layer_id];
        lb.total += (size_t) m.expert_bytes;
        if (m.role == llama_expert_tensor_role::DOWN) {
            lb.has_down = true;
        } else {
            lb.has_up = true;
        }
    }

    if (p->max_expert_size == 0) {
        LLAMA_LOG_ERROR("nexus-expert-stream: no supported MoE expert tensors found\n");
        result.status = llama_expert_create_status::NO_SUPPORTED_TENSORS;
        return result;
    }

    p->pair_bytes = 0;
    if (capacity_known && !per_layer.empty()) {
        for (const auto & kv : per_layer) {
            if (!kv.second.has_down || !kv.second.has_up) {
                // una capa sin ambas mitades: el par no esta identificado
                capacity_known = false;
                break;
            }
            p->pair_bytes = std::max(p->pair_bytes, kv.second.total);
        }
    }
    if (!capacity_known) {
        p->pair_bytes = 0;
        LLAMA_LOG_INFO("nexus-expert-stream: expert pair size unknown (tensor role/layer not provided), "
                       "L2 capacity in pairs not reported\n");
    }

    p->arena_bytes = cfg.ram_warm_bytes;
    p->alignment   = cfg.alignment > 0 ? cfg.alignment : 64;

    // Cota de cordura: una arena RAM por encima de 1 TiB o de tamano cero no
    // puede ser una configuracion valida del cache. Se rechaza ANTES de llamar
    // al allocator para no provocar un fallo de tamano en el propio malloc.
    if (p->arena_bytes == 0 || p->arena_bytes > ((size_t) 1 << 40)) {
        LLAMA_LOG_ERROR("nexus-expert-stream: unreasonable arena size requested: %zu\n", p->arena_bytes);
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }

    // Inicializar reserva host independiente con posix_memalign
    constexpr size_t MAX_ALIGN = ((size_t) 1) << (std::numeric_limits<size_t>::digits - 1);
    if (p->alignment == 0 || (p->alignment & (p->alignment - 1)) != 0 ||
        p->alignment % sizeof(void *) != 0 || p->alignment > MAX_ALIGN) {
        LLAMA_LOG_ERROR("nexus-expert-stream: invalid alignment requested: %zu\n", p->alignment);
        result.status = llama_expert_create_status::IO_ERROR;
        return result;
    }

    int res = posix_memalign(&p->arena_base, p->alignment, p->arena_bytes);
    if (res != 0 || !p->arena_base) {
        LLAMA_LOG_ERROR("nexus-expert-stream: failed to allocate L2 arena (%zu bytes): %s\n",
            p->arena_bytes, strerror(res));
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }

    p->owns_arena = true;
    p->arena_buf  = nullptr;
    p->mem_kind   = llama_expert_mem_kind::PAGEABLE_HOST;

    // fase asignadora 3: nodo inicial de free_holes
    if (nexus_test_fail_alloc()) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }
    p->free_holes[0] = p->arena_bytes;
    p->aligned_bytes = 0;
    p->slack_bytes   = 0;
    p->free_bytes    = p->arena_bytes;

    LLAMA_LOG_INFO(
        "nexus-expert-stream: L2 single arena initialized: %.2f MiB (alignment: %zu, first-fit coalescing)\n",
        (double) p->arena_bytes / (1024.0 * 1024.0), p->alignment);

    // fase asignadora 4: objeto fuente
    if (nexus_test_fail_alloc()) {
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    }
    result.status = llama_expert_create_status::SUCCESS;
    result.source = std::make_unique<llama_expert_source>();
    result.source->pimpl = std::move(p);
    return result;
    } catch (const std::bad_alloc &) {
        // p (unique_ptr) se destruye y cierra los shards ya abiertos: no quedan
        // recursos colgados. Un fallo de memoria nunca se clasifica como IO_ERROR.
        result.status = llama_expert_create_status::OUT_OF_MEMORY;
        return result;
    } catch (...) {
        result.status = llama_expert_create_status::IO_ERROR;
        return result;
    }
}

llama_expert_source::~llama_expert_source() = default;

llama_expert_fill_status llama_expert_source::fill_expert_impl(
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
    if (it_ext == p.extents.end()) {
        return llama_expert_fill_status::FALLBACK_ALLOWED;
    }
    if (it_ext->second.validation == llama_expert_tensor_validation::INVALID) {
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    if (it_ext->second.validation == llama_expert_tensor_validation::UNSUPPORTED &&
        !it_ext->second.is_supported) {
        return llama_expert_fill_status::NOT_SUPPORTED;
    }
    // Los fixtures/consumidores antiguos pueden omitir validation y conservar
    // is_supported=true; se acepta esa compatibilidad solo para esos metadatos.

    const auto & ext = it_ext->second;

    // B3: validar los metadatos ANTES de calcular offsets. pread_full() no puede
    // detectar un offset que ya se desbordo al calcularlo, y un extent invalido
    // nunca debe convertirse en una lectura mmap silenciosa.
    if (ext.shard_idx < 0 || (size_t) ext.shard_idx >= p.shards.size()) {
        LLAMA_LOG_ERROR("nexus-expert-stream: invalid shard index %d for %s\n",
                ext.shard_idx, ggml_get_name(w));
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    const auto & shard = p.shards[ext.shard_idx];
    if (shard.fd < 0) {
        LLAMA_LOG_ERROR("nexus-expert-stream: shard %s is not open\n", shard.path.c_str());
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    if (expert < 0 || ext.n_expert <= 0 || expert >= ext.n_expert) {
        LLAMA_LOG_ERROR("nexus-expert-stream: expert %d out of range [0, %d) for %s\n",
                expert, ext.n_expert, ggml_get_name(w));
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    if (ext.expert_bytes == 0 || ext.expert_stride < ext.expert_bytes) {
        LLAMA_LOG_ERROR("nexus-expert-stream: invalid expert layout for %s (bytes %" PRIu64 ", stride %" PRIu64 ")\n",
                ggml_get_name(w), ext.expert_bytes, ext.expert_stride);
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    if (expert_size != (size_t) ext.expert_bytes) {
        LLAMA_LOG_ERROR("nexus-expert-stream: expert size %zu does not match the extent %" PRIu64 " of %s\n",
                expert_size, ext.expert_bytes, ggml_get_name(w));
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    // expert * expert_stride + base_offset sin desbordar 64 bits
    if ((uint64_t) expert > (UINT64_MAX - ext.base_offset) / ext.expert_stride) {
        LLAMA_LOG_ERROR("nexus-expert-stream: offset overflow for %s (expert %d)\n", ggml_get_name(w), expert);
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    const uint64_t file_offset = ext.base_offset + (uint64_t) expert * ext.expert_stride;
    if (file_offset > shard.size || (uint64_t) expert_size > shard.size - file_offset) {
        LLAMA_LOG_ERROR("nexus-expert-stream: extent of %s (expert %d) is outside shard %s\n",
                ggml_get_name(w), expert, shard.path.c_str());
        return llama_expert_fill_status::INVALID_EXTENT;
    }
    if (backend != nullptr) {
        if (bank == nullptr) {
            LLAMA_LOG_ERROR("nexus-expert-stream: no bank for %s on a backend transfer\n", ggml_get_name(w));
            return llama_expert_fill_status::INVALID_EXTENT;
        }
        const size_t bank_bytes = ggml_nbytes(bank);
        if (slot_offset > bank_bytes || expert_size > bank_bytes - slot_offset) {
            LLAMA_LOG_ERROR("nexus-expert-stream: slot of %s does not fit the bank (%zu + %zu > %zu)\n",
                    ggml_get_name(w), slot_offset, expert_size, bank_bytes);
            return llama_expert_fill_status::INVALID_EXTENT;
        }
    }

    if (expert_size > p.arena_bytes) {
        return llama_expert_fill_status::NOT_SUPPORTED;
    }

    // --- 1+2. Coordinación de carga (bloqueante A) ---
    // La búsqueda, la reclamación de la clave y la inserción del bloque LOADING
    // ocurren en una única sección crítica. Un segundo hilo solo puede observar
    // RESIDENT o LOADING; si observa LOADING espera en cv_loading sin reclamar
    // la clave, de modo que una clave tiene exactamente un cargador.
    bool   is_loader   = false;
    size_t blk_offset  = 0;
    size_t alloc_off   = 0;
    size_t alloc_sz    = 0;
    size_t alloc_slk   = 0;
    const llama_expert_key key { il, reinterpret_cast<uintptr_t>(w), expert };

    for (;;) {
        std::unique_lock<std::mutex> lk(p.l2_mutex);
        if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
            return llama_expert_fill_status::IO_ERROR;
        }

        auto it = p.l2_index.find(key);
        if (it != p.l2_index.end()) {
            if (it->second.content == llama_expert_content::RESIDENT) {
                p.stats.l2_hits++;
                nexus_l2_block & blk = it->second;
                blk.pin_layers++;
                blk.lru_tick = ++p.current_lru_tick;
                blk_offset = blk.offset;

                p.l2_lru_list.erase(blk.lru_it);
                p.l2_lru_list.push_front(key);
                blk.lru_it = p.l2_lru_list.begin();
                break;
            }
            if (it->second.content == llama_expert_content::LOADING) {
                // Otro hilo está cargando esta clave: esperar en cv_loading. No
                // se reclama la clave ni se incrementan misses.
                p.cv_loading.wait(lk, [&]{
                    return p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed) ||
                        p.l2_index.find(key) == p.l2_index.end() ||
                        p.l2_index[key].content != llama_expert_content::LOADING;
                });
                continue;
            }
            // INVALID o estado anómalo
            return llama_expert_fill_status::IO_ERROR;
        }

        // Clave ausente: solo este hilo puede reclamarla ahora (mutex tomado).
        // Miss contabilizado bajo el mutex; nadie más cuenta esta clave.
        p.stats.l2_misses++;
        is_loader = true;

        while (!p.alloc_block(expert_size, alloc_off, alloc_sz, alloc_slk)) {
            if (!p.evict_lru_victim()) {
                lk.unlock();
                p.synchronize_active_fill();
                lk.lock();

                if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
                    return llama_expert_fill_status::IO_ERROR;
                }
                if (!p.evict_lru_victim()) {
                    p.alloc_fail++;
                    p.oom_rejects++;
                    return llama_expert_fill_status::OUT_OF_MEMORY;
                }
            }
        }

        // B3: el destino completo debe caber en la arena. El allocator no
        // deberia entregar un rango fuera de ella; si lo hiciera, se
        // devuelve el hueco y se rechaza el extent en vez de escribir fuera.
        if (alloc_off > p.arena_bytes || alloc_sz > p.arena_bytes - alloc_off ||
            expert_size > alloc_sz) {
            p.alloc_fail++;
            if (!p.free_block(alloc_off, alloc_sz, alloc_slk)) {
                return llama_expert_fill_status::OUT_OF_MEMORY;
            }
            return llama_expert_fill_status::INVALID_EXTENT;
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

        try {
            p.l2_lru_list.push_front(key);
            blk.lru_it = p.l2_lru_list.begin();
            const auto inserted = p.l2_index.emplace(key, blk);
            if (!inserted.second) {
                // Inalcanzable bajo el mismo mutex (la clave no existia al mirar),
                // pero se mantiene como red de seguridad: restaurar y continuar.
                p.l2_lru_list.erase(blk.lru_it);
                p.alloc_fail++;
                if (!p.free_block(alloc_off, alloc_sz, alloc_slk)) {
                    return llama_expert_fill_status::OUT_OF_MEMORY;
                }
                is_loader = false;
                lk.unlock();
                continue;
            }
        } catch (const std::bad_alloc &) {
            // El hueco y los contadores se restauran antes de propagar al
            // wrapper C-safe; no queda una clave parcialmente publicada.
            if (!p.l2_lru_list.empty() && p.l2_lru_list.front().layer_id == key.layer_id &&
                p.l2_lru_list.front().tensor_id == key.tensor_id &&
                p.l2_lru_list.front().expert_id == key.expert_id) {
                p.l2_lru_list.pop_front();
            }
            p.free_block(alloc_off, alloc_sz, alloc_slk);
            return llama_expert_fill_status::OUT_OF_MEMORY;
        } catch (...) {
            // El hueco y los contadores se restauran antes de propagar al
            // wrapper C-safe; no queda una clave parcialmente publicada.
            if (!p.l2_lru_list.empty() && p.l2_lru_list.front().layer_id == key.layer_id &&
                p.l2_lru_list.front().tensor_id == key.tensor_id &&
                p.l2_lru_list.front().expert_id == key.expert_id) {
                p.l2_lru_list.pop_front();
            }
            p.free_block(alloc_off, alloc_sz, alloc_slk);
            return llama_expert_fill_status::OUT_OF_MEMORY;
        }
        blk_offset = alloc_off;
        break;
    }

    // --- 3. Carga NVMe (solo el hilo que reclamo la clave) ---
    if (is_loader) {
        // Mientras el bloque existe (LOADING/RESIDENT), la arena no puede
        // liberarse ni sustituirse. Un hilo que llegue a una barrera durante
        // esta ventana se pone a la espera; nunca ve la arena sin dueño.
        {
            std::unique_lock<std::mutex> lk(p.l2_mutex);
            if (p.teardown || p.aborted || p.fatal.load(std::memory_order_relaxed)) {
                return llama_expert_fill_status::IO_ERROR;
            }
        }
        void * dst_ptr = static_cast<uint8_t *>(p.arena_base) + blk_offset;

        const auto t0 = std::chrono::steady_clock::now();
        int64_t n_read = llama_expert_pread_full(shard.fd, dst_ptr, expert_size, file_offset, shard.size);
        const auto t1 = std::chrono::steady_clock::now();

        if (n_read != (int64_t) expert_size) {
            LLAMA_LOG_ERROR("nexus-expert-stream: pread_full failed for %s exp %d\n", ggml_get_name(w), expert);
            {
                std::lock_guard<std::mutex> lk(p.l2_mutex);
                auto it = p.l2_index.find(key);
                if (it != p.l2_index.end()) {
                    if (!p.free_block(it->second.offset, it->second.size, it->second.slack)) {
                        // No se pudo devolver el hueco (OOM): el bloque se
                        // conserva marcado INVALID para que los esperadores no lo
                        // reutilicen, y la contabilidad sigue siendo coherente.
                        it->second.content = llama_expert_content::INVALID;
                        p.cv_loading.notify_all();
                        return llama_expert_fill_status::IO_ERROR;
                    }
                    p.l2_lru_list.erase(it->second.lru_it);
                    p.l2_index.erase(it);
                }
                p.cv_loading.notify_all(); // B2: Despertar cualquier hilo que esperaba
            }
            return llama_expert_fill_status::IO_ERROR;
        }

        const uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        p.stats.proc_bytes += expert_size;
        p.stats.completed_reads.fetch_add(1, std::memory_order_relaxed);
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

        // La transferencia solo puede programarse si el bloque RESIDENT sigue
        // existiendo y coincide exactamente con el rango que se va a leer.
        // Registrar primero el inflight protege la arena frente a eviction y
        // permite que la barrera lo confirme por epoch.
        auto it = p.l2_index.find(key);
        if (it == p.l2_index.end() ||
            it->second.content != llama_expert_content::RESIDENT ||
            it->second.offset != blk_offset ||
            it->second.req_bytes != expert_size ||
            it->second.offset > p.arena_bytes ||
            it->second.req_bytes > p.arena_bytes - it->second.offset) {
            LLAMA_LOG_ERROR("nexus-expert-stream: source block disappeared or changed before upload\n");
            return llama_expert_fill_status::IO_ERROR;
        }

        // A partir de aquí la hebra programa una transferencia: cuenta para la
        // quiescencia de las barreras hasta que regrese de set_async.
        ++p.programming_fills;
        guard.arm_programming();

        if (p.fill_epoch == UINT64_MAX) {
            p.enter_fail_stop(fail_reason::epoch_overflow);
        }
        const uint64_t next_epoch = p.fill_epoch + 1;
        try {
            p.l2_inflight.push_back({ key, next_epoch });
        } catch (...) {
            // No se modifico el bloque ni el epoch: el guard libera
            // programming_fills y no queda ninguna transferencia anunciada.
            return llama_expert_fill_status::OUT_OF_MEMORY;
        }
        p.fill_epoch = next_epoch;
        it->second.inflight++;
        it->second.last_fill_epoch = next_epoch;

        lk.unlock();

        void * src_ptr = static_cast<uint8_t *>(p.arena_base) + blk_offset;
        try {
            ggml_backend_tensor_set_async(backend, bank, src_ptr, slot_offset, expert_size);
        } catch (...) {
            // La interfaz C de GGML no debe recibir una excepción. El registro
            // ya protege el bloque; conservarlo como pendiente permite que el
            // teardown lo trate como no confirmado, en vez de reutilizar la arena.
            p.fatal.store(true, std::memory_order_release);
            return llama_expert_fill_status::IO_ERROR;
        }
        p.stats.upload_bytes.fetch_add(expert_size, std::memory_order_relaxed);
        p.stats.uploads_programmed.fetch_add(1, std::memory_order_relaxed);
    }

    return llama_expert_fill_status::LOADED;
}

llama_expert_fill_status llama_expert_source::fill_expert(
        ggml_backend_t backend,
        ggml_tensor * bank,
        int32_t il,
        const ggml_tensor * w,
        int32_t expert,
        size_t slot_offset,
        size_t expert_size) {
    llama_expert_fill_status status = llama_expert_fill_status::IO_ERROR;
    try {
        status = fill_expert_impl(backend, bank, il, w, expert, slot_offset, expert_size);
    } catch (...) {
        // Nunca dejar que una excepción C++ atraviese el callback C del scheduler.
        pimpl->fatal.store(true, std::memory_order_release);
        status = llama_expert_fill_status::IO_ERROR;
    }
    pimpl->stats.fill_attempts.fetch_add(1, std::memory_order_relaxed);
    if (status == llama_expert_fill_status::LOADED) {
        pimpl->stats.fills.fetch_add(1, std::memory_order_relaxed);
    } else {
        pimpl->stats.record(status);
    }
    return status;
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
    const size_t pair_sz = p.pair_bytes;
    if (pair_sz == 0) {
        // capacidad desconocida: no se inventa una estimacion
        return 0;
    }
    return p.arena_bytes / pair_sz;
}

size_t llama_expert_source::get_l2_block_count() const {
    std::lock_guard<std::mutex> lk(pimpl->l2_mutex);
    return pimpl->l2_index.size();
}

size_t llama_expert_source::get_effective_alignment() const {
    return pimpl->alignment;
}

size_t llama_expert_source::get_arena_bytes() const {
    return pimpl->arena_bytes;
}

llama_expert_mem_kind llama_expert_source::get_l2_mem_kind() const {
    return pimpl->mem_kind;
}
