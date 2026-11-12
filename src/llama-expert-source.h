#pragma once

#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct ggml_tensor;

// ---------------------------------------------------------------------------
// NEXUS D-0 - llama_expert_source
//
// Arena L2 unica de tamano variable que REEMPLAZA al staging ring.  Una sola
// reserva de `config::ram_warm_bytes` es a la vez cache caliente y origen de
// las transferencias hacia VRAM.  No hay `ring_slots` ni `stage_used` ni copia
// entre dos reservas (P-01).
//
// Identidad de un bloque: SOLO (layer_id, tensor_id, expert_id).  `tensor_id`
// es el puntero (uintptr_t) del `ggml_tensor *`.  No hay heuristica por tamano
// ni por nombre ni por "arena_type" (R-08).
//
// Cada bloque tiene TRES ejes ortogonales independientes (correccion A):
//   - content   : datos, como valor EMPTY | LOADING | RESIDENT | INVALID
//   - pin_layers: contador de capas vivas que lo tienen pinneado (>= 0)
//   - inflight  : contador de transferencias programadas no confirmadas (>= 0)
// Ademas, solo en Ruta 2 forzada, `needs_proof`.
// ---------------------------------------------------------------------------

enum class llama_expert_fill_status {
    LOADED,            // Experto transferido al buffer de VRAM exitosamente
    FALLBACK_ALLOWED,  // Tensor excluido de streaming (permitido leer de w->data)
    NOT_SUPPORTED,     // Layout/geometria incompatible con streaming (p. ej. experto > arena)
    IO_ERROR,          // Error fisico de lectura o pread corrupto
    INVALID_EXTENT,    // Desbordamiento de limites de archivo o metadatos invalidos
    OUT_OF_MEMORY      // Memoria insuficiente para L2 tras expulsar
};

// contenido de los datos de un bloque, independiente de pin/inflight
enum class llama_expert_content {
    EMPTY,     // no hay bits validos para la clave (hueco / nunca cargado)
    LOADING,   // copy programada o en curso (puede valer RESIDENT, pero no confirmado)
    RESIDENT,  // los bytes del experto estan en la arena
    INVALID    // el `synchronize` que cubria la ultima programacion fallo: no se puede probar
};

// clave de identidad de una asignacion en la arena L2 (solo estos tres campos)
struct llama_expert_key {
    int32_t   layer_id  = -1;
    uintptr_t tensor_id = 0; // reinterpret_cast<uintptr_t>(w)
    int32_t   expert_id = -1;

    bool operator==(const llama_expert_key & o) const {
        return layer_id == o.layer_id && tensor_id == o.tensor_id && expert_id == o.expert_id;
    }
    bool operator!=(const llama_expert_key & o) const { return !(*this == o); }
};

struct llama_expert_key_hash {
    size_t operator()(const llama_expert_key & k) const;
};

// clasificacion del modo efectivo de un bloque tras la reserva (§2.6)
enum class llama_expert_mem_kind {
    PINNED_VK,              // el buft host obtenido del backend es el efectivo
    PAGEABLE_HOST,          // posix_memalign directo (sin backend)
    PAGEABLE_HOST_FALLBACK, // el backend no expone host buft: posix_memalign con WARN
};

// Estado resumido de un bloque L2 (para pruebas e inspeccion, no para el hot path).
struct llama_expert_block_info {
    llama_expert_content content    = llama_expert_content::EMPTY;
    int32_t              pin_layers = 0;
    int32_t              inflight   = 0;
    bool                 needs_proof = false;
    uint64_t             lru_tick   = 0;
    size_t               offset     = 0;
    size_t               size       = 0;
    size_t               slack      = 0;
    llama_expert_mem_kind mem_kind  = llama_expert_mem_kind::PAGEABLE_HOST;
};

struct llama_expert_arena_stats {
    // contabilidad FISICA de la arena. invariant:
    //   aligned_bytes + slack_bytes + free_bytes == arena_bytes
    // (NO existe una metrica agregada `used`)
    std::atomic<uint64_t> arena_bytes    { 0 }; // tamano total reservado
    std::atomic<uint64_t> aligned_bytes  { 0 }; // suma de `size` de asignaciones vivas
    std::atomic<uint64_t> slack_bytes    { 0 }; // suma de `slack` (`0 <= slack < A`) de asignaciones vivas
    std::atomic<uint64_t> free_bytes     { 0 }; // suma de huecos gestionables

    // desgloses informativos (no entran en el invariante)
    std::atomic<uint64_t> bytes_loading     { 0 }; // residuo con contenido LOADING
    std::atomic<uint64_t> largest_free_hole { 0 }; // mayor hueco contiguo (para mayor_hueco_acusable)

    std::atomic<uint64_t> blocks_live     { 0 };
    std::atomic<uint64_t> blocks_resident { 0 };
    std::atomic<uint64_t> blocks_invalid  { 0 };
    std::atomic<uint64_t> blocks_pinned   { 0 };
    std::atomic<uint64_t> blocks_inflight { 0 };

    std::atomic<uint64_t> align_a         { 0 }; // alineacion efectiva del buft

    std::atomic<uint64_t> evictions       { 0 }; // expulsiones contiguas realizadas
    std::atomic<uint64_t> evict_bytes     { 0 }; // bytes vividos expulsados (Σ size)
    std::atomic<uint64_t> evict_slack_bytes { 0 }; // slack desperdiciado por las expulsiones

    std::atomic<uint64_t> sync_calls      { 0 }; // `ggml_backend_synchronize` invocados por la arena
    std::atomic<uint64_t> sync_failures   { 0 }; // de esos, que lanzaron

    std::atomic<uint64_t> alloc_fail      { 0 };
    std::atomic<uint64_t> oom_rejects     { 0 };
};

struct llama_expert_stats {
    std::atomic<uint64_t> fills          { 0 };
    std::atomic<uint64_t> l2_hits        { 0 };
    std::atomic<uint64_t> l2_misses      { 0 };
    std::atomic<uint64_t> proc_bytes     { 0 };
    std::atomic<uint64_t> io_us_total    { 0 };
    std::atomic<uint64_t> io_us_max      { 0 };
    std::atomic<uint64_t> drops          { 0 };
    std::atomic<uint64_t> verify_ok      { 0 };
    std::atomic<uint64_t> verify_fail    { 0 };

    void log(const char * tag) const;
    void reset();
};

struct llama_expert_file_meta {
    std::string path;
    uint64_t    size = 0;
};

// Rol explicito de un tensor de expertos dentro de la capa MoE. La identidad
// gate/up vs down NUNCA se infiere por tamano ni por nombre parcial.
enum class llama_expert_tensor_role {
    UNKNOWN, // sin rol explicito: no se puede componer el par de la capa
    GATE,
    UP,
    GATE_UP, // tensor fusionado gate+up
    DOWN
};

struct llama_expert_tensor_meta {
    int32_t  shard_idx     = -1;
    uint64_t base_offset   = 0;
    uint64_t expert_stride = 0; // t->nb[2] real
    uint64_t expert_bytes  = 0;
    int32_t  n_expert      = 0;
    bool     is_supported  = false;
    int32_t  layer_id      = -1; // capa a la que pertenece, -1 desconocida
    llama_expert_tensor_role role = llama_expert_tensor_role::UNKNOWN;
};

struct llama_expert_model_meta {
    std::vector<llama_expert_file_meta> files;
    std::unordered_map<const ggml_tensor *, llama_expert_tensor_meta> tensors;
};

// Funcion canonica para lectura robusta desde almacenamiento
int64_t llama_expert_pread_full(int fd, void * buf, size_t count, uint64_t offset, uint64_t file_size);

class llama_expert_source {
public:
    struct config {
        // arena L2 unica de tamano variable (reemplaza ram_warm/ring separados)
        size_t ram_warm_bytes = 1024ull * 1024 * 1024;
        size_t alignment      = 0;                  // 0 = ggml_backend_buft_get_alignment del buft host
        bool   stats          = false;
        bool   verify         = false;
        bool   path_2         = false;              // forzar Ruta 2 (needs_proof) [experimental]
    };

    static std::unique_ptr<llama_expert_source> create(
        const llama_expert_model_meta & meta,
        const config & cfg);

    ~llama_expert_source();

    // Rellena el experto faltante en el banco de VRAM.
    // — `backend == nullptr` suprime la programacion (modo CPU/unitarias).
    llama_expert_fill_status fill_expert(
        ggml_backend_t backend,
        ggml_tensor * bank,
        int32_t il,
        const ggml_tensor * w,
        int32_t expert,
        size_t slot_offset,
        size_t expert_size);

    // Se invoca cuando una capa previa ha sido confirmada como sincronizada en el backend.
    // SOLO decrementa `pin_layers` (correccion A). No demuestra fin de transferencia.
    void on_layer_synchronized(int32_t il);

    // Sincronizacion explicita final (para ultima capa o destruccion).
    // Ejecuta el contrato de seis pasos de §2.5.1.
    void synchronize_all();

    // Detiene la admision de fills, espera a los fills activos y confirma las
    // transferencias programadas antes de que el llamante libere la arena.
    // Idempotente; el destructor lo invoca. Si no puede demostrar que una
    // transferencia pendiente termino (barrera activa o sincronizacion fallida),
    // la arena se retira sin liberarse (cuarentena del proceso) en vez de
    // devolver memoria que el dispositivo podria estar usando.
    void drain();

    // true si la arena se retiro sin liberarse por no poder demostrar que el
    // dispositivo dejo de usarla (inspeccion para pruebas)
    bool was_arena_quarantined() const;

    // Confirmacion en dos fases de las transferencias programadas en el backend
    // enlazado, cuando la sincronizacion la hace el consumidor (p. ej. el MoE
    // cache antes de leer el slot map):
    //   begin_sync(): cierra la compuerta y espera a que ninguna hebra este
    //                 programando una transferencia; toma una instantanea del epoch.
    //   <el llamante ejecuta ggml_backend_synchronize() en el backend enlazado>
    //   end_sync()  : confirma las transferencias cubiertas por esa sincronizacion
    //                 y reabre la compuerta.
    // Si begin_sync() no puede cerrar la compuerta (teardown/abort/fail-stop),
    // end_sync() no confirma nada.
    void begin_sync();
    void end_sync();

    const llama_expert_stats & get_stats() const;
    const llama_expert_arena_stats & get_arena_stats() const;

    // metadatos / inspeccion para pruebas unitarias
    bool get_block_info(const llama_expert_key & key, llama_expert_block_info & out) const;
    size_t get_l2_capacity_pairs() const; // desde los roles/capas explicitos; 0 = desconocida, nunca fijado a 320
    size_t get_l2_block_count() const;
    size_t get_arena_bytes() const;
    llama_expert_mem_kind get_l2_mem_kind() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
