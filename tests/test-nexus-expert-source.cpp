#include "../src/llama-expert-source.h"
#include "../ggml/include/ggml-alloc.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST FAILURE at %s:%d: %s\n", __FILE__, __LINE__, msg); \
            abort(); \
        } \
    } while (0)

static void test_d16_alloc_failure_injection() {
    printf("[TEST] Testing D-1.6 deterministic allocation-failure injection...\n");

    // --- 1. Eviction con fallo de liberacion: free_block devuelve false y el
    // bloque victimizado se conserva integro (indices, LRU, contadores).
    {
        llama_expert_model_meta meta;
        llama_expert_source::config cfg;
        cfg.ram_warm_bytes = 4 * 1024 * 1024;
        cfg.alignment      = 64;

        char temp_path[] = "/tmp/test_nexus_d16_XXXXXX";
        int fd = mkstemp(temp_path);
        TEST_CHECK(fd >= 0, "mkstemp failed");
        const size_t file_size = 16 * 1024 * 1024;
        TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");
        meta.files.push_back({ temp_path, file_size });

        ggml_tensor t{};
        t.ne[2] = 8;
        t.nb[2] = 1024 * 1024;
        meta.tensors[&t] = { 0, 0, 1024 * 1024, 1024 * 1024, 8, true };

        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create failed");
        ggml_tensor dummy_bank{};

        // Llenar la arena por completo: 4 expertos de 1 MiB, sin huecos.
        for (int e = 0; e < 4; ++e) {
            auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, e, 0, 1024 * 1024);
            TEST_CHECK(st == llama_expert_fill_status::LOADED, "arena fill failed");
        }
        // Liberar los pins de la capa 0 para que los bloques sean evictables.
        exsrc->on_layer_synchronized(0);
        TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 4, "arena must be full");

        // La arena esta llena (sin huecos): la eviction del bloque LRU-rear
        // pasa por el caso de "hueco nuevo" de free_block. Armar el modo
        // "fallar siempre" hace que TODOS los intentos de liberacion fallen
        // (el bucle de la ruta de miss reintenta una vez tras la barrera), de
        // modo que la eviction aborta y el bloque se conserva.
        llama_expert_source::test_arm_alloc_failure(-2);
        auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, 4, 0, 1024 * 1024);
        llama_expert_source::test_disarm_alloc_failure();

        TEST_CHECK(st == llama_expert_fill_status::OUT_OF_MEMORY,
                   "a failed eviction must surface as OUT_OF_MEMORY");
        TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 4,
                   "the failed eviction must not despublicar a block");
        TEST_CHECK(exsrc->get_arena_stats().evictions.load() == 0,
                   "no eviction may be counted when the free failed");

        llama_expert_key victim{ 0, reinterpret_cast<uintptr_t>(&t), 0 };
        llama_expert_block_info bi{};
        TEST_CHECK(exsrc->get_block_info(victim, bi),
                   "the LRU-rear victim must still be indexed");
        const auto & s = exsrc->get_arena_stats();
        TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
                   "invariant must hold after the failed eviction");

        // El allocator sigue usable: una nueva llamada evicta de verdad.
        st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, 4, 0, 1024 * 1024);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "the allocator must keep working");
        TEST_CHECK(exsrc->get_arena_stats().evictions.load() == 1, "the real eviction must be counted");

        exsrc->drain();
        exsrc.reset();
        close(fd);
        unlink(temp_path);
    }

    // --- 2. create(): barrido de presupuesto. Cada fase asignadora, cuando
    // falla, debe devolver OUT_OF_MEMORY (nunca lanzar, nunca INVALID/IO para
    // metadatos validos) y, cuando llega a SUCCESS, la fuente debe funcionar.
    {
        llama_expert_model_meta meta;
        char temp_path[] = "/tmp/test_nexus_d16c_XXXXXX";
        int fd = mkstemp(temp_path);
        TEST_CHECK(fd >= 0, "mkstemp failed");
        const size_t file_size = 16 * 1024 * 1024;
        TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");
        meta.files.push_back({ temp_path, file_size });

        ggml_tensor t{};
        t.ne[2] = 8;
        t.nb[2] = 1024 * 1024;
        meta.tensors[&t] = { 0, 0, 1024 * 1024, 1024 * 1024, 8, true,
                             0, llama_expert_tensor_role::GATE_UP,
                             llama_expert_tensor_validation::VALID };

        llama_expert_source::config cfg;
        cfg.ram_warm_bytes = 4 * 1024 * 1024;
        cfg.alignment      = 64;

        bool saw_oom  = false;
        bool saw_ok   = false;
        // Hay 5 fases asignadoras en create(); un presupuesto >= 5 deja pasar
        // todas y debe devolver SUCCESS.
        for (long long phase = 0; phase <= 8; ++phase) {
            llama_expert_source::test_arm_alloc_failure(phase);
            auto res = llama_expert_source::create(meta, cfg);
            llama_expert_source::test_disarm_alloc_failure();

            if (res.status == llama_expert_create_status::OUT_OF_MEMORY) {
                saw_oom = true;
                TEST_CHECK(res.source == nullptr, "no source may exist on OOM");
            } else if (res.status == llama_expert_create_status::SUCCESS) {
                saw_ok = true;
                TEST_CHECK(res.source != nullptr, "successful create must own a source");
                ggml_tensor dummy_bank{};
                auto st = res.source->fill_expert(nullptr, &dummy_bank, 0, &t, 0, 0, 1024 * 1024);
                TEST_CHECK(st == llama_expert_fill_status::LOADED,
                           "a source built under a tight budget must still work");
                res.source->drain();
                res.source.reset();
            } else {
                TEST_CHECK(false, "create must only report OOM or SUCCESS under allocation failures");
            }
        }
        TEST_CHECK(saw_oom, "some allocation phase must fail within the budget range");
        TEST_CHECK(saw_ok, "a sufficient budget must succeed");

        close(fd);
        unlink(temp_path);
    }

    printf("[PASS] D-1.6 allocation-failure injection passed.\n");
}

static void test_pread_full() {
    printf("[TEST] Testing llama_expert_pread_full robustness...\n");

    char temp_path[] = "/tmp/test_nexus_pread_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "Failed to create temp file");

    // Escribir 64 KiB con un patrón determinista
    const size_t file_size = 65536;
    std::vector<uint8_t> expected_data(file_size);
    for (size_t i = 0; i < file_size; ++i) {
        expected_data[i] = (uint8_t)(i % 251);
    }
    ssize_t written = write(fd, expected_data.data(), file_size);
    TEST_CHECK(written == (ssize_t)file_size, "write failed");

    // 1. Lectura completa exacta
    std::vector<uint8_t> read_buf(file_size);
    int64_t res = llama_expert_pread_full(fd, read_buf.data(), file_size, 0, file_size);
    TEST_CHECK(res == (int64_t)file_size, "pread_full did not return full file size");
    TEST_CHECK(memcmp(read_buf.data(), expected_data.data(), file_size) == 0, "Data read mismatch");

    // 2. Lectura parcial en el medio
    std::vector<uint8_t> chunk(1024);
    res = llama_expert_pread_full(fd, chunk.data(), 1024, 2048, file_size);
    TEST_CHECK(res == 1024, "Partial pread_full failed");
    TEST_CHECK(memcmp(chunk.data(), expected_data.data() + 2048, 1024) == 0, "Partial data mismatch");

    // 3. Offset fuera de límites del archivo (debe retornar -ERANGE)
    res = llama_expert_pread_full(fd, chunk.data(), 1024, file_size + 1, file_size);
    TEST_CHECK(res == -ERANGE, "Offset out of bounds did not return -ERANGE");

    // 4. Lectura que excede el tamaño del archivo (debe retornar -ERANGE)
    res = llama_expert_pread_full(fd, chunk.data(), 1024, file_size - 512, file_size);
    TEST_CHECK(res == -ERANGE, "Read exceeding file size did not return -ERANGE");

    // 5. Overflow aritmético de 64 bits (debe retornar -EOVERFLOW)
    res = llama_expert_pread_full(fd, chunk.data(), 1024, UINT64_MAX - 500, file_size);
    TEST_CHECK(res == -EOVERFLOW, "Integer overflow did not return -EOVERFLOW");

    // 6. EOF prematuro: truncar archivo a 512 bytes e intentar leer 1024 con file_size falso
    int tr_res = ftruncate(fd, 512);
    TEST_CHECK(tr_res == 0, "ftruncate failed");
    res = llama_expert_pread_full(fd, chunk.data(), 1024, 0, 2048);
    TEST_CHECK(res == -EIO, "Premature EOF did not return -EIO");

    close(fd);
    unlink(temp_path);
    printf("[PASS] llama_expert_pread_full passed all invariant tests.\n");
}

static void test_l2_dimensioning_and_pinning() {
    printf("[TEST] Testing L2 cache dimensioning and pinning invariants...\n");

    char temp_path[] = "/tmp/test_nexus_model_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "Failed to create temp file");

    // Tamaño de experto de Gemma 4 Q4_0:
    // gate_up: 2,230,272 bytes
    // down:    1,115,136 bytes
    // total par: 3,345,408 bytes
    const size_t sz_gate_up = 2230272;
    const size_t sz_down    = 1115136;

    // Crear un archivo de prueba con varios bloques
    const size_t fake_file_size = 50 * 1024 * 1024;
    int tr_res = ftruncate(fd, fake_file_size);
    TEST_CHECK(tr_res == 0, "ftruncate failed");

    llama_expert_model_meta meta;
    meta.files.push_back({ temp_path, fake_file_size });

    // Tensores mock
    ggml_tensor t_gate_up{};
    t_gate_up.name[0] = 'g';
    t_gate_up.ne[0] = 2816;
    t_gate_up.ne[1] = 1408;
    t_gate_up.ne[2] = 128;
    t_gate_up.nb[2] = sz_gate_up;

    ggml_tensor t_down{};
    t_down.name[0] = 'd';
    t_down.ne[0] = 704;
    t_down.ne[1] = 2816;
    t_down.ne[2] = 128;
    t_down.nb[2] = sz_down;

    meta.tensors[&t_gate_up] = { 0, 0, sz_gate_up, sz_gate_up, 128, true, 0, llama_expert_tensor_role::GATE_UP };
    meta.tensors[&t_down]    = { 0, sz_gate_up * 10, sz_down, sz_down, 128, true, 0, llama_expert_tensor_role::DOWN };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 1024ull * 1024 * 1024; // 1024 MiB

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "Failed to create llama_expert_source");

    // Condición de diseño D-0: capacidad en pares para la arena única
    size_t l2_pairs = exsrc->get_l2_capacity_pairs();
    printf("  L2 capacity in pairs: %zu (Expected: ~320)\n", l2_pairs);
    TEST_CHECK(l2_pairs == (cfg.ram_warm_bytes / (sz_gate_up + sz_down)), "L2 capacity mismatch");

    // Simular dummy fill_expert sobre un buffer falso (sin backend activo)
    ggml_tensor dummy_bank{};
    llama_expert_fill_status status = exsrc->fill_expert(
        nullptr, &dummy_bank, 0, &t_gate_up, 0, 0, sz_gate_up);
    TEST_CHECK(status == llama_expert_fill_status::LOADED, "fill_expert failed on cold miss");

    // Segunda llamada al mismo experto -> DEBE ser un HIT en L2
    status = exsrc->fill_expert(
        nullptr, &dummy_bank, 0, &t_gate_up, 0, 0, sz_gate_up);
    TEST_CHECK(status == llama_expert_fill_status::LOADED, "fill_expert failed on warm hit");

    const auto & stats = exsrc->get_stats();
    TEST_CHECK(stats.fills == 2, "fills stat count mismatch");
    TEST_CHECK(stats.fill_attempts == 2, "fill_attempts stat count mismatch");
    TEST_CHECK(stats.l2_hits == 1, "l2_hits stat count mismatch");
    TEST_CHECK(stats.l2_misses == 1, "l2_misses stat count mismatch");

    // Sincronizar capa 0: desmarca los pins de capa 0
    exsrc->on_layer_synchronized(0);

    close(fd);
    unlink(temp_path);
    printf("[PASS] L2 dimensioning and pinning passed all invariant tests.\n");
}


static void test_allocator_coalescing_and_fragmentation() {
    printf("[TEST] Testing first-fit coalescing and physical slack invariants...\n");

    llama_expert_model_meta meta;
    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 10 * 1024 * 1024; // 10 MiB
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_frag_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "Failed to create temp file");
    const size_t fake_file_size = 20 * 1024 * 1024;
    ftruncate(fd, fake_file_size);
    meta.files.push_back({ temp_path, fake_file_size });

    ggml_tensor t1{}, t2{}, t3{}, t_big{};
    t1.ne[2] = 10; t1.nb[2] = 1000000;
    t2.ne[2] = 10; t2.nb[2] = 2000000;
    t3.ne[2] = 10; t3.nb[2] = 3000000;
    t_big.ne[2] = 5; t_big.nb[2] = 7000000;

    meta.tensors[&t1]    = { 0, 0, 1000000, 1000000, 10, true };
    meta.tensors[&t2]    = { 0, 1000000, 2000000, 2000000, 10, true };
    meta.tensors[&t3]    = { 0, 3000000, 3000000, 3000000, 10, true };
    meta.tensors[&t_big] = { 0, 6000000, 7000000, 7000000, 5, true };

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "Create failed");

    ggml_tensor dummy_bank{};
    // Alloc 1 (1,000,000 bytes)
    exsrc->fill_expert(nullptr, &dummy_bank, 0, &t1, 0, 0, 1000000);
    // Alloc 2 (2,000,000 bytes)
    exsrc->fill_expert(nullptr, &dummy_bank, 0, &t2, 0, 0, 2000000);
    // Alloc 3 (3,000,000 bytes)
    exsrc->fill_expert(nullptr, &dummy_bank, 0, &t3, 0, 0, 3000000);

    const auto & s = exsrc->get_arena_stats();
    printf("  Allocated: %lu, Slack: %lu, Free: %lu, Total: %lu\n",
           s.aligned_bytes.load(), s.slack_bytes.load(), s.free_bytes.load(), s.arena_bytes.load());
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "Invariant violated during multiple allocs");

    // Liberar pines de capa 0
    exsrc->on_layer_synchronized(0);

    // Debe provocar evictions de los 3 bloques y fusionar sus huecos contiguos mediante coalescing
    auto status = exsrc->fill_expert(nullptr, &dummy_bank, 1, &t_big, 0, 0, 7000000);
    TEST_CHECK(status == llama_expert_fill_status::LOADED, "Big alloc after eviction and coalescing failed");

    const auto & s_post = exsrc->get_arena_stats();
    printf("  Post-Eviction: Evictions=%lu, EvictBytes=%lu\n",
           s_post.evictions.load(), s_post.evict_bytes.load());
    TEST_CHECK(s_post.evictions.load() >= 3, "Not all victim blocks were evicted");
    TEST_CHECK(s_post.aligned_bytes.load() + s_post.slack_bytes.load() + s_post.free_bytes.load() == s_post.arena_bytes.load(),
               "Invariant violated after eviction and coalescing");

    close(fd);
    unlink(temp_path);
    printf("[PASS] Allocator first-fit, slack and coalescing passed all invariant tests.\n");
}

static void test_concurrency_and_barrier_quiescence() {
    printf("[TEST] Testing concurrent fills and barrier quiescence...\n");

    llama_expert_model_meta meta;
    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 20 * 1024 * 1024; // 20 MiB
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_conc_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "Failed to create temp file");
    const size_t fake_file_size = 20 * 1024 * 1024;
    ftruncate(fd, fake_file_size);
    meta.files.push_back({ temp_path, fake_file_size });

    ggml_tensor t_c1{}, t_c2{}, t_c3{};
    t_c1.ne[2] = 10; t_c1.nb[2] = 1000000;
    t_c2.ne[2] = 10; t_c2.nb[2] = 1000000;
    t_c3.ne[2] = 10; t_c3.nb[2] = 1000000;

    meta.tensors[&t_c1] = { 0, 0, 1000000, 1000000, 10, true };
    meta.tensors[&t_c2] = { 0, 2000000, 1000000, 1000000, 10, true };
    meta.tensors[&t_c3] = { 0, 4000000, 1000000, 1000000, 10, true };

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "Create failed");

    ggml_tensor dummy_bank{};

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> workers;

    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&, i]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int step = 0; step < 10; ++step) {
                ggml_tensor * target = (i % 3 == 0) ? &t_c1 : ((i % 3 == 1) ? &t_c2 : &t_c3);
                exsrc->fill_expert(nullptr, &dummy_bank, i % 2, target, step % 5, 0, 1000000);
                if (step % 4 == 0) {
                    exsrc->on_layer_synchronized(i % 2);
                }
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto & w : workers) {
        w.join();
    }

    exsrc->synchronize_all();

    const auto & s = exsrc->get_arena_stats();
    printf("  Concurrent Arena Stats: Live=%lu, Resident=%lu, SyncCalls=%lu\n",
           s.blocks_live.load(), s.blocks_resident.load(), s.sync_calls.load());
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "Invariant violated after concurrent stress");

    close(fd);
    unlink(temp_path);
    printf("[PASS] Concurrent fills and barrier quiescence passed all invariant tests.\n");
}


static void test_blocker_a_concurrent_same_key() {
    printf("[TEST] Testing blocker A: concurrent loads of the same key...\n");

    // 1. 2, 4 y 8 hilos pidiendo la MISMA clave (0, t, 0). Una unica lectura.
    for (int n_threads : { 2, 4, 8 }) {
        llama_expert_model_meta meta;
        llama_expert_source::config cfg;
        cfg.ram_warm_bytes = 4 * 1024 * 1024;
        cfg.alignment      = 64;

        char temp_path[] = "/tmp/test_nexus_a_XXXXXX";
        int fd = mkstemp(temp_path);
        TEST_CHECK(fd >= 0, "mkstemp failed");
        const size_t file_size = 8 * 1024 * 1024;
        TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");
        meta.files.push_back({ temp_path, file_size });

        ggml_tensor t{};
        t.ne[2] = 8;
        t.nb[2] = 1024 * 1024;
        meta.tensors[&t] = { 0, 0, 1024 * 1024, 1024 * 1024, 8, true };

        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create failed");

        ggml_tensor dummy_bank{};

        // Barrera: los hilos se alinean DENTRO de la seccion de consulta de la
        // clave, cuando ya miraron el estado. Uno solo podra reclamarla.
        std::atomic<int> reached{0};
        std::atomic<bool> release{false};
        std::atomic<int> loaded{0};
        std::atomic<int> failed{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, i]() {
                reached.fetch_add(1, std::memory_order_release);
                while (!release.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                const auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, 0, 0, 1024 * 1024);
                if (st == llama_expert_fill_status::LOADED) {
                    loaded.fetch_add(1, std::memory_order_release);
                } else {
                    failed.fetch_add(1, std::memory_order_release);
                }
            });
        }
        while (reached.load(std::memory_order_acquire) != n_threads) {
            std::this_thread::yield();
        }
        // Dormir un instante para maximizar la probabilidad de que todos hayan
        // observado el estado de la clave antes de que el primero la reclame.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        release.store(true, std::memory_order_release);
        for (auto & w : workers) {
            w.join();
        }

        TEST_CHECK(loaded.load() == n_threads && failed.load() == 0,
                   "all waiters must observe the loader's RESIDENT block");
        TEST_CHECK(exsrc->get_stats().l2_misses.load() == 1,
                   "exactly one NVMe read per key is allowed");
        TEST_CHECK(exsrc->get_stats().completed_reads.load() == 1,
                   "completed reads must match the single miss");

        // No quedan entradas LOADING ni bloques perdidos.
        TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 1,
                   "exactly one block must remain");
        TEST_CHECK(exsrc->get_arena_stats().blocks_resident.load() == 1,
                   "the remaining block must be RESIDENT");
        const auto & s = exsrc->get_arena_stats();
        TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
                   "arena invariant violated");

        exsrc->drain();
        exsrc.reset();
        close(fd);
        unlink(temp_path);
    }

    // 2. Fallo de lectura: todos los solicitantes se despiertan y un nuevo
    // intento posterior puede volver a cargar la clave.
    {
        llama_expert_model_meta meta;
        llama_expert_source::config cfg;
        cfg.ram_warm_bytes = 2 * 1024 * 1024;
        cfg.alignment      = 64;

        char temp_path[] = "/tmp/test_nexus_a_fail_XXXXXX";
        int fd = mkstemp(temp_path);
        TEST_CHECK(fd >= 0, "mkstemp failed");
        const size_t file_size = 4 * 1024 * 1024;
        TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");
        meta.files.push_back({ temp_path, file_size });

        ggml_tensor t{};
        t.ne[2] = 8;
        t.nb[2] = 1024 * 1024;
        // Extent dentro del archivo pero se truncara debajo antes de la lectura.
        meta.tensors[&t] = { 0, 0, 1024 * 1024, 1024 * 1024, 8, true };

        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create failed");
        ggml_tensor dummy_bank{};

        std::atomic<int> reached{0};
        std::atomic<bool> release{false};
        std::atomic<int> io_errors{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < 4; ++i) {
            workers.emplace_back([&]() {
                reached.fetch_add(1, std::memory_order_release);
                while (!release.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                const auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, 0, 0, 1024 * 1024);
                if (st == llama_expert_fill_status::IO_ERROR) {
                    io_errors.fetch_add(1, std::memory_order_release);
                }
            });
        }
        while (reached.load(std::memory_order_acquire) != 4) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        TEST_CHECK(ftruncate(fd, 512 * 1024) == 0, "ftruncate failed");
        release.store(true, std::memory_order_release);
        for (auto & w : workers) {
            w.join();
        }
        // Todos observaron el fallo (el que leyo y los que esperaban).
        TEST_CHECK(io_errors.load() == 4, "every waiter must see the read failure");
        TEST_CHECK(exsrc->get_stats().completed_reads.load() == 0,
                   "a failed read must never count as completed");
        TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 0,
                   "no block must survive a failed read");

        // Cada hilo despertado reintenta la clave y, como el archivo sigue
        // truncado, cada reintento es un miss que falla. No puede quedar una
        // entrada LOADING tras todos los reintentos.
        TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 0,
                   "no LOADING entry may survive after the notifications");

        // Un nuevo intento posterior puede volver a cargar la clave.
        TEST_CHECK(ftruncate(fd, 4 * 1024 * 1024) == 0, "ftruncate restore failed");
        auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t, 0, 0, 1024 * 1024);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "the key must be loadable again");
        TEST_CHECK(exsrc->get_stats().completed_reads.load() == 1,
                   "the retry must complete exactly one read");
        TEST_CHECK(exsrc->get_arena_stats().blocks_resident.load() == 1,
                   "the retry must leave a RESIDENT block");

        exsrc->drain();
        exsrc.reset();
        close(fd);
        unlink(temp_path);
    }

    printf("[PASS] Blocker A concurrent same-key loads passed.\n");
}


static void test_blocker_b_create_statuses() {
    printf("[TEST] Testing blocker B: explicit create outcomes...\n");

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 4 * 1024 * 1024;
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_b_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t file_size = 16 * 1024 * 1024;
    TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");

    // 1. Sin candidatos MoE -> NO_SUPPORTED_TENSORS (fallback mmap permitido).
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::NO_SUPPORTED_TENSORS,
                   "empty metadata must be NO_SUPPORTED_TENSORS");
        TEST_CHECK(res.source == nullptr, "no source may exist on failure");
    }

    // 2. Tensor valido + tensor UNSUPPORTED -> SUCCESS (el valido streama).
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor ok{}, unsup{};
        meta.tensors[&ok]    = { 0, 0,       1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::VALID };
        meta.tensors[&unsup] = { 0, 8000000, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::DOWN,    llama_expert_tensor_validation::UNSUPPORTED };
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::SUCCESS,
                   "a valid tensor plus an unsupported one must succeed");
        TEST_CHECK(res.source != nullptr, "source must exist");
        res.source->drain();
        res.source.reset();
    }

    // 3. Todos los candidatos INVALID -> INVALID_METADATA (error, no fallback).
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor bad1{}, bad2{};
        meta.tensors[&bad1] = { 0, 0,       1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::INVALID };
        meta.tensors[&bad2] = { 0, 8000000, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::DOWN,    llama_expert_tensor_validation::INVALID };
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::INVALID_METADATA,
                   "all-INVALID metadata must be INVALID_METADATA");
        TEST_CHECK(res.source == nullptr, "no source may exist on invalid metadata");
    }

    // 4. Shard inexistente -> IO_ERROR.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ "/nonexistent/nexus_shard.bin", file_size });
        ggml_tensor t{};
        meta.tensors[&t] = { 0, 0, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::VALID };
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::IO_ERROR,
                   "an unopenable shard must be IO_ERROR");
    }

    // 5. Indice de shard fuera de rango -> INVALID_METADATA.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor t{};
        meta.tensors[&t] = { 7, 0, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::VALID };
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::INVALID_METADATA,
                   "an out-of-range shard index must be INVALID_METADATA");
    }

    // 6. Extent fuera del archivo -> INVALID_METADATA.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor t{};
        meta.tensors[&t] = { 0, file_size - 4096, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::VALID };
        auto res = llama_expert_source::create(meta, cfg);
        TEST_CHECK(res.status == llama_expert_create_status::INVALID_METADATA,
                   "an extent beyond the file must be INVALID_METADATA");
    }

    // 7. Reserva de arena imposible -> OUT_OF_MEMORY.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor t{};
        meta.tensors[&t] = { 0, 0, 1000000, 1000000, 4, true, 0, llama_expert_tensor_role::GATE_UP, llama_expert_tensor_validation::VALID };
        llama_expert_source::config huge;
        huge.ram_warm_bytes = (size_t) -4;
        huge.alignment      = 64;
        auto res = llama_expert_source::create(meta, huge);
        TEST_CHECK(res.status == llama_expert_create_status::OUT_OF_MEMORY,
                   "an impossible arena reservation must be OUT_OF_MEMORY");
    }

    close(fd);
    unlink(temp_path);
    printf("[PASS] Blocker B explicit create outcomes passed.\n");
}


static void test_b1_b4_robustness_invariants() {
    printf("[TEST] Testing Bloques B1-B4 robustness invariants...\n");

    llama_expert_model_meta meta;
    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 10 * 1024 * 1024;
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_b1b4_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t fake_file_size = 10 * 1024 * 1024;
    ftruncate(fd, fake_file_size);
    meta.files.push_back({ temp_path, fake_file_size });

    ggml_tensor t_m1{};
    t_m1.ne[2] = 5; t_m1.nb[2] = 1000000;
    meta.tensors[&t_m1] = { 0, 0, 1000000, 1000000, 5, true };

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "Create failed");

    // 1. Test B3: Backend Mismatch
    ggml_backend_t cpu_b1 = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cpu_b2 = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(cpu_b1 != nullptr && cpu_b2 != nullptr, "Failed to initialize CPU backends");

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu_b1));
    const size_t bank_total_bytes = 5000000;
    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(cpu_buft, bank_total_bytes);
    TEST_CHECK(bank_buf != nullptr, "Failed to alloc bank buffer");

    ggml_tensor dummy_bank{};
    dummy_bank.type = GGML_TYPE_I8;
    dummy_bank.ne[0] = bank_total_bytes;
    dummy_bank.ne[1] = 1;
    dummy_bank.ne[2] = 1;
    dummy_bank.ne[3] = 1;
    dummy_bank.nb[0] = 1;
    dummy_bank.nb[1] = bank_total_bytes;
    dummy_bank.nb[2] = bank_total_bytes;
    dummy_bank.nb[3] = bank_total_bytes;
    dummy_bank.buffer = bank_buf;
    dummy_bank.data = ggml_backend_buffer_get_base(bank_buf);

    // Primer fill enlaza cpu_b1
    auto st1 = exsrc->fill_expert(cpu_b1, &dummy_bank, 0, &t_m1, 0, 0, 1000000);
    TEST_CHECK(st1 == llama_expert_fill_status::LOADED, "First fill on cpu_b1 failed");

    // Segundo fill con cpu_b2 -> DEBE rechazar con IO_ERROR (B3)
    auto st2 = exsrc->fill_expert(cpu_b2, &dummy_bank, 0, &t_m1, 1, 0, 1000000);
    TEST_CHECK(st2 == llama_expert_fill_status::IO_ERROR, "Backend mismatch did not return IO_ERROR");

    // 2. Test B2: Múltiples hilos compitiendo por exactamente el mismo experto
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            auto s = exsrc->fill_expert(cpu_b1, &dummy_bank, 0, &t_m1, 2, i * 1000000, 1000000);
            if (s == llama_expert_fill_status::LOADED) {
                success_count++;
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    TEST_CHECK(success_count.load() == 4, "Concurrent same-key fill failed");

    // 3. Test B1: Destrucción limpia y segura
    exsrc.reset();
    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(cpu_b1);
    ggml_backend_free(cpu_b2);

    close(fd);
    unlink(temp_path);
    printf("[PASS] Bloques B1-B4 robustness invariants passed all tests.\n");
}

static void test_two_phase_sync_and_teardown() {
    printf("[TEST] Testing begin_sync/end_sync, pread failure and teardown invariants...\n");

    llama_expert_model_meta meta;
    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 10 * 1024 * 1024;
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_sync_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t fake_file_size = 10 * 1024 * 1024;
    ftruncate(fd, fake_file_size);
    meta.files.push_back({ temp_path, fake_file_size });

    ggml_tensor t_ok{}, t_bad{};
    t_ok.ne[2]  = 5; t_ok.nb[2]  = 1000000;
    t_bad.ne[2] = 5; t_bad.nb[2] = 1000000;
    meta.tensors[&t_ok]  = { 0, 0, 1000000, 1000000, 5, true };
    // t_bad sera truncado por debajo de su extent durante la prueba: el archivo
    // se encoge DESPUES de crear la fuente para producir un IO_ERROR fisico.
    meta.tensors[&t_bad] = { 0, 4000000, 1000000, 1000000, 5, true };

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "Create failed");

    // Antes de cualquier fill no hay nada programado: begin/end no cuentan sync
    exsrc->begin_sync();
    exsrc->end_sync();
    TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
               "empty begin/end must not count a synchronization");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(cpu != nullptr, "Failed to initialize CPU backend");
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu));
    const size_t bank_total_bytes = 5000000;
    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(cpu_buft, bank_total_bytes);
    TEST_CHECK(bank_buf != nullptr, "Failed to alloc bank buffer");

    ggml_tensor bank{};
    bank.type = GGML_TYPE_I8;
    bank.ne[0] = bank_total_bytes;
    bank.ne[1] = 1;
    bank.ne[2] = 1;
    bank.ne[3] = 1;
    bank.nb[0] = 1;
    bank.nb[1] = bank_total_bytes;
    bank.nb[2] = bank_total_bytes;
    bank.nb[3] = bank_total_bytes;
    bank.buffer = bank_buf;
    bank.data   = ggml_backend_buffer_get_base(bank_buf);

    auto st = exsrc->fill_expert(cpu, &bank, 0, &t_ok, 0, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill on CPU backend failed");
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
               "a programmed transfer must be inflight until confirmed");

    // Confirmacion en dos fases: el llamante sincroniza entre begin y end
    exsrc->begin_sync();
    ggml_backend_synchronize(cpu);
    exsrc->end_sync();
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 0,
               "end_sync must confirm the covered transfer");
    TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 1,
               "end_sync must count exactly one synchronization");

    // La compuerta quedo abierta: un nuevo fill debe poder programarse
    st = exsrc->fill_expert(cpu, &bank, 0, &t_ok, 1, 1000000, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill after end_sync failed");

    // end_sync sin begin_sync no confirma nada
    exsrc->end_sync();
    TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 1,
               "end_sync without begin_sync must be a no-op");
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
               "end_sync without begin_sync must not confirm the transfer");

    // t_bad tiene un extent valido: se carga normal.
    st = exsrc->fill_expert(cpu, &bank, 0, &t_bad, 0, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "a valid extent must load");

    // Error fisico de lectura: el archivo se trunca por debajo de un extent que
    // los metadatos declaraban valido. pread_full falla y se propaga IO_ERROR.
    TEST_CHECK(ftruncate(fd, 512 * 1024) == 0, "ftruncate failed");
    st = exsrc->fill_expert(cpu, &bank, 0, &t_ok, 2, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::IO_ERROR, "pread failure must return IO_ERROR");

    // restaurar el tamano para las pruebas siguientes
    TEST_CHECK(ftruncate(fd, fake_file_size) == 0, "ftruncate restore failed");

    const auto & s = exsrc->get_arena_stats();
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "arena invariant violated after a failed read");

    // Destruccion con una transferencia aun pendiente: drain_for_teardown la
    // confirma antes de liberar la arena.
    exsrc.reset();

    // Destruccion sin backend enlazado (modo HOST_ONLY)
    auto exsrc_host = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc_host != nullptr, "Create (HOST_ONLY) failed");
    ggml_tensor dummy_bank{};
    st = exsrc_host->fill_expert(nullptr, &dummy_bank, 0, &t_ok, 0, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "HOST_ONLY fill failed");
    exsrc_host.reset();

    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(cpu);

    close(fd);
    unlink(temp_path);
    printf("[PASS] Two-phase sync, pread failure and teardown passed all tests.\n");
}

static ggml_tensor make_test_bank(ggml_backend_buffer_t buf, size_t bytes) {
    ggml_tensor bank{};
    bank.type = GGML_TYPE_I8;
    bank.ne[0] = bytes;
    bank.ne[1] = 1;
    bank.ne[2] = 1;
    bank.ne[3] = 1;
    bank.nb[0] = 1;
    bank.nb[1] = bytes;
    bank.nb[2] = bytes;
    bank.nb[3] = bytes;
    bank.buffer = buf;
    bank.data   = ggml_backend_buffer_get_base(buf);
    return bank;
}

static void test_b2_sync_modes_and_teardown() {
    printf("[TEST] Testing B2 sync modes, quiescence and teardown policies...\n");

    llama_expert_model_meta meta;
    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 2 * 1024 * 1024; // dos bloques de ~1 MB
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_b2_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t fake_file_size = 16 * 1024 * 1024;
    TEST_CHECK(ftruncate(fd, fake_file_size) == 0, "ftruncate failed");
    meta.files.push_back({ temp_path, fake_file_size });

    ggml_tensor t_a{}, t_b{};
    t_a.ne[2] = 4; t_a.nb[2] = 1000000;
    t_b.ne[2] = 4; t_b.nb[2] = 1000000;
    meta.tensors[&t_a] = { 0, 0,       1000000, 1000000, 4, true };
    meta.tensors[&t_b] = { 0, 8000000, 1000000, 1000000, 4, true };

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(cpu != nullptr, "failed to init the CPU backend");
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu));
    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(cpu_buft, 2 * 1000000);
    TEST_CHECK(bank_buf != nullptr, "failed to allocate the bank");
    ggml_tensor bank = make_test_bank(bank_buf, 2 * 1000000);

    // --- 1. UNKNOWN: una fuente sin ningun fill no tiene transferencias que
    // confirmar; ni synchronize_all() ni drain() deben inventar un fallo.
    {
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (UNKNOWN) failed");
        exsrc->synchronize_all();
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
                   "a source with no transfers must not count a synchronization");
        exsrc->drain();
        TEST_CHECK(!exsrc->was_arena_quarantined(),
                   "drain without transfers must not quarantine the arena");
        TEST_CHECK(exsrc->get_arena_stats().sync_failures.load() == 0,
                   "drain without transfers must not register a synchronization failure");
        exsrc.reset();
    }

    // --- 2. HOST_ONLY: los fills no programan transferencias.
    {
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (HOST_ONLY) failed");
        ggml_tensor dummy_bank{};
        auto st = exsrc->fill_expert(nullptr, &dummy_bank, 0, &t_a, 0, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "HOST_ONLY fill failed");
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 0,
                   "HOST_ONLY must not program a transfer");
        exsrc->synchronize_all();
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
                   "HOST_ONLY synchronize_all must not call the backend");
        exsrc->drain();
        TEST_CHECK(!exsrc->was_arena_quarantined(),
                   "HOST_ONLY drain must not quarantine the arena");
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
                   "HOST_ONLY drain must not call the backend");
        exsrc.reset();
    }

    // --- 3. CPU_SYNC: un bloque con transferencia pendiente no se reutiliza.
    // Con la arena llena de expertos no confirmados y sus pins ya soltados, el
    // siguiente fill solo puede tener exito si sincroniza antes de expulsar.
    {
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (CPU_SYNC) failed");

        auto st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 0, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill 0 failed");
        st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 1, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill 1 failed");
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 2,
                   "both transfers must be inflight before the synchronization");
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
                   "no synchronization must be counted yet");

        // los pins se sueltan, pero las transferencias no estan confirmadas
        exsrc->on_layer_synchronized(0);

        st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 2, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill 2 after eviction failed");
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 1,
                   "the allocator must synchronize before reusing an unconfirmed block");
        TEST_CHECK(exsrc->get_arena_stats().evictions.load() == 1,
                   "exactly one block must be evicted");
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
                   "only the new transfer may remain inflight");

        exsrc->drain();
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 0,
                   "drain must confirm the last pending transfer");
        TEST_CHECK(!exsrc->was_arena_quarantined(),
                   "drain after a successful synchronization must free the arena");
        st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 3, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::IO_ERROR,
                   "a fill after drain must be rejected");
        exsrc.reset();
    }

    // --- 4. Teardown con barrera activa (begin_sync sin end_sync): no se puede
    // demostrar que el dispositivo dejo de leer la arena -> cuarentena.
    {
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (barrier) failed");
        auto st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 0, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill before the barrier failed");

        exsrc->begin_sync(); // cierra la compuerta; el llamante no sincroniza
        exsrc->drain();
        TEST_CHECK(exsrc->was_arena_quarantined(),
                   "drain with an active barrier must not free the arena");
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
                   "drain with an active barrier must not fake a confirmation");
        TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 0,
                   "drain with an active barrier must not call the backend");

        exsrc->end_sync(); // drain ya limpio sync_pending: no confirma nada
        TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
                   "end_sync after drain must not confirm anything");
        exsrc.reset(); // la arena esta en cuarentena: no se libera
    }

    // --- 5. Un fill que espera en la compuerta cuando empieza el teardown.
    {
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (gate wait) failed");
        auto st = exsrc->fill_expert(cpu, &bank, 0, &t_a, 0, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill before the gate failed");

        exsrc->begin_sync(); // compuerta cerrada

        std::atomic<bool> about_to_fill{false};
        llama_expert_fill_status worker_status = llama_expert_fill_status::LOADED;
        std::thread worker([&]() {
            about_to_fill.store(true, std::memory_order_release);
            worker_status = exsrc->fill_expert(cpu, &bank, 0, &t_a, 1, 0, 1000000);
        });
        while (!about_to_fill.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        exsrc->drain(); // cierra la compuerta y despierta al worker
        worker.join();

        TEST_CHECK(worker_status == llama_expert_fill_status::IO_ERROR,
                   "a fill blocked on the gate must be rejected by the teardown");
        TEST_CHECK(exsrc->was_arena_quarantined(),
                   "drain with an active barrier must quarantine the arena");
        exsrc.reset();
    }

    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(cpu);
    close(fd);
    unlink(temp_path);
    printf("[PASS] B2 sync modes, quiescence and teardown policies passed.\n");
}

static void test_b3_metadata_validation() {
    printf("[TEST] Testing B3 metadata and offset validation...\n");

    char temp_path[] = "/tmp/test_nexus_b3_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t file_size = 8 * 1024 * 1024;
    TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");

    llama_expert_model_meta meta;
    meta.files.push_back({ temp_path, file_size });

    // Los metadatos corruptos (shard fuera de rango, stride < bytes, overflow,
    // extent fuera del archivo) se rechazan YA en create(): los cubre
    // test_blocker_b_create_statuses. Aqui se usa un unico tensor valido y se
    // ejercita la validacion por argumentos de fill_expert.
    ggml_tensor t_ok{};

    meta.tensors[&t_ok] = { 0, 0, 1000000, 1000000, 4, true };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 8 * 1024 * 1024;
    cfg.alignment      = 64;

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "create failed");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(cpu != nullptr, "failed to init the CPU backend");
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu));

    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(cpu_buft, 1000000);
    TEST_CHECK(bank_buf != nullptr, "failed to allocate the bank");
    ggml_tensor bank = make_test_bank(bank_buf, 1000000);

    ggml_backend_buffer_t small_buf = ggml_backend_buft_alloc_buffer(cpu_buft, 4096);
    TEST_CHECK(small_buf != nullptr, "failed to allocate the small bank");
    ggml_tensor small_bank = make_test_bank(small_buf, 4096);

    auto expect_invalid = [&](const ggml_tensor * t, ggml_tensor * b, int32_t expert, size_t size) {
        const auto st = exsrc->fill_expert(cpu, b, 0, t, expert, 0, size);
        TEST_CHECK(st == llama_expert_fill_status::INVALID_EXTENT,
                   "invalid metadata must return INVALID_EXTENT");
    };

    expect_invalid(&t_ok,            &bank, -1, 1000000); // experto negativo
    expect_invalid(&t_ok,            &bank, 4, 1000000);  // experto fuera de [0, n_expert)
    expect_invalid(&t_ok,            &bank, 0, 999999);   // expert_size != extent
    expect_invalid(&t_ok,            &small_bank, 0, 1000000); // no cabe en el banco

    // backend sin banco: no puede programarse la transferencia
    {
        const auto st = exsrc->fill_expert(cpu, nullptr, 0, &t_ok, 0, 0, 1000000);
        TEST_CHECK(st == llama_expert_fill_status::INVALID_EXTENT,
                   "a transfer without a bank must return INVALID_EXTENT");
    }

    // ningun rechazo debe haber leido del archivo ni dejado bloques vivos
    TEST_CHECK(exsrc->get_stats().proc_bytes.load() == 0,
               "a rejected extent must not read from the file");
    TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 0,
               "a rejected extent must not leave a live block");
    const auto & s = exsrc->get_arena_stats();
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "arena invariant violated after rejected extents");

    // un extent valido sigue funcionando
    auto st = exsrc->fill_expert(cpu, &bank, 0, &t_ok, 3, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "a valid extent must still load");

    exsrc->drain();
    TEST_CHECK(!exsrc->was_arena_quarantined(), "drain after B3 must free the arena");
    exsrc.reset();

    ggml_backend_buffer_free(small_buf);
    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(cpu);
    close(fd);
    unlink(temp_path);
    printf("[PASS] B3 metadata and offset validation passed.\n");
}

static void test_b4_tensor_identity() {
    printf("[TEST] Testing B4 explicit tensor identity (no size heuristic)...\n");

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 20 * 1024 * 1024;
    cfg.alignment      = 64;

    char temp_path[] = "/tmp/test_nexus_b4_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t file_size = 64 * 1024 * 1024;
    TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");

    // 1. Sin rol ni capa: la capacidad en pares es desconocida, aunque los
    // tamanos "parezcan" gate_up/down.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor big{}, small{};
        meta.tensors[&big]   = { 0, 0,        4000000, 4000000, 8, true };
        meta.tensors[&small] = { 0, 32000000, 1000000, 1000000, 8, true };
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (unknown roles) failed");
        TEST_CHECK(exsrc->get_l2_capacity_pairs() == 0,
                   "a size threshold must not identify gate/up vs down");
        exsrc->drain();
        exsrc.reset();
    }

    // 2. Rol explicito con un gate_up pequeno: la capacidad se calcula por rol,
    // no por el umbral de tamano.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor gate_up{}, down{};
        meta.tensors[&gate_up] = { 0, 0,       1000000, 1000000, 8, true, 0, llama_expert_tensor_role::GATE_UP };
        meta.tensors[&down]    = { 0, 8000000, 2000000, 2000000, 8, true, 0, llama_expert_tensor_role::DOWN };
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (explicit roles) failed");
        TEST_CHECK(exsrc->get_l2_capacity_pairs() == cfg.ram_warm_bytes / 3000000,
                   "capacity must use the explicit roles, not the sizes");
        exsrc->drain();
        exsrc.reset();
    }

    // 3. gate + up separados: el par es gate + up + down.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor gate{}, up{}, down{};
        meta.tensors[&gate] = { 0, 0,        2000000, 2000000, 8, true, 3, llama_expert_tensor_role::GATE };
        meta.tensors[&up]   = { 0, 16000000, 2000000, 2000000, 8, true, 3, llama_expert_tensor_role::UP   };
        meta.tensors[&down] = { 0, 32000000, 1000000, 1000000, 8, true, 3, llama_expert_tensor_role::DOWN };
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (gate+up) failed");
        TEST_CHECK(exsrc->get_l2_capacity_pairs() == cfg.ram_warm_bytes / 5000000,
                   "a separate gate+up pair must include both tensors");
        exsrc->drain();
        exsrc.reset();
    }

    // 4. Una capa sin la mitad down (o sin gate/up) no identifica el par.
    {
        llama_expert_model_meta meta;
        meta.files.push_back({ temp_path, file_size });
        ggml_tensor gate_up{}, down_other{};
        meta.tensors[&gate_up]    = { 0, 0,        2000000, 2000000, 8, true, 1, llama_expert_tensor_role::GATE_UP };
        meta.tensors[&down_other] = { 0, 16000000, 1000000, 1000000, 8, true, 2, llama_expert_tensor_role::DOWN };
        auto exsrc = llama_expert_source::create(meta, cfg).source;
        TEST_CHECK(exsrc != nullptr, "create (split layers) failed");
        TEST_CHECK(exsrc->get_l2_capacity_pairs() == 0,
                   "a layer without both halves must not report a capacity");
        exsrc->drain();
        exsrc.reset();
    }

    close(fd);
    unlink(temp_path);
    printf("[PASS] B4 explicit tensor identity passed.\n");
}

static void test_b5_statistics() {
    printf("[TEST] Testing B5 statistics accounting and reset...\n");

    llama_expert_model_meta meta;
    char temp_path[] = "/tmp/test_nexus_b5_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t expert_bytes = 1024 * 1024;
    TEST_CHECK(ftruncate(fd, 4 * expert_bytes) == 0, "ftruncate failed");
    meta.files.push_back({ temp_path, 4 * expert_bytes });

    ggml_tensor t{};
    t.ne[2] = 4;
    t.nb[2] = expert_bytes;
    meta.tensors[&t] = { 0, 0, expert_bytes, expert_bytes, 4, true };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 2 * expert_bytes;
    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "create failed");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(cpu != nullptr, "failed to init CPU backend");
    auto buft = ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu));
    auto bank_buf = ggml_backend_buft_alloc_buffer(buft, expert_bytes);
    TEST_CHECK(bank_buf != nullptr, "failed to allocate bank");
    ggml_tensor bank = make_test_bank(bank_buf, expert_bytes);

    auto st = exsrc->fill_expert(cpu, &bank, 0, &t, 0, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "first statistics fill failed");
    st = exsrc->fill_expert(cpu, &bank, 0, &t, 0, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "L2 hit statistics fill failed");
    exsrc->begin_sync();
    ggml_backend_synchronize(cpu);
    exsrc->end_sync();

    const auto & stats = exsrc->get_stats();
    auto & mutable_stats = const_cast<llama_expert_stats &>(stats);
    TEST_CHECK(stats.fills.load() == 2, "fills must count successful fills");
    TEST_CHECK(stats.l2_misses.load() == 1 && stats.l2_hits.load() == 1,
               "L2 hit/miss counters must be separated");
    TEST_CHECK(stats.proc_bytes.load() == expert_bytes,
               "proc_bytes must count only the NVMe read");
    TEST_CHECK(stats.upload_bytes.load() == 2 * expert_bytes,
               "upload_bytes must count both device uploads");
    (void) stats.io_us_total.load(); // contador legible, de tipo unsigned

    // Los contadores de rutas alternativas se prueban directamente para que el
    // test no dependa de fabricar mmap/capacidad insuficiente en el fixture.
    mutable_stats.record(llama_expert_fill_status::FALLBACK_ALLOWED);
    mutable_stats.record(llama_expert_fill_status::NOT_SUPPORTED);
    mutable_stats.record(llama_expert_fill_status::OUT_OF_MEMORY);
    mutable_stats.record(llama_expert_fill_status::IO_ERROR);
    mutable_stats.record(llama_expert_fill_status::INVALID_EXTENT);
    TEST_CHECK(stats.fallback_mmap.load() == 1 && stats.not_supported.load() == 1 &&
               stats.out_of_memory.load() == 1 && stats.rejected_io.load() == 1 &&
               stats.rejected_invalid.load() == 1,
               "rejected/fallback routes must have separate counters");

    mutable_stats.reset();
    TEST_CHECK(stats.fills.load() == 0 && stats.l2_hits.load() == 0 &&
               stats.upload_bytes.load() == 0 && stats.rejected_invalid.load() == 0,
               "reset must clear every B5 counter");

    exsrc->drain();
    exsrc.reset();
    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(cpu);
    close(fd);
    unlink(temp_path);
    printf("[PASS] B5 statistics accounting and reset passed.\n");
}

static void test_blocker_c_alignment_transaction() {
    printf("[TEST] Testing blocker C/D: arena alignment substitution is transactional...\n");

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (gpu_dev == nullptr) {
        printf("[SKIP] no GPU/iGPU backend: arena substitution not exercised in this build\n");
        return;
    }
    ggml_backend_t vk = ggml_backend_dev_init(gpu_dev, nullptr);
    TEST_CHECK(vk != nullptr, "failed to init the GPU backend");
    const char * vk_name = ggml_backend_name(vk);
    if (vk_name == nullptr || strstr(vk_name, "Vulkan") == nullptr) {
        printf("[SKIP] the GPU backend is not Vulkan (%s)\n", vk_name ? vk_name : "unknown");
        ggml_backend_free(vk);
        return;
    }

    const size_t expert_bytes = 1024 * 1024;
    char temp_path[] = "/tmp/test_nexus_c_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    const size_t file_size = 4 * expert_bytes;
    TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");

    llama_expert_model_meta meta;
    meta.files.push_back({ temp_path, file_size });
    ggml_tensor t{};
    t.ne[2] = 4;
    t.nb[2] = expert_bytes;
    meta.tensors[&t] = { 0, 0, expert_bytes, expert_bytes, 4, true };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 2 * expert_bytes;
    cfg.alignment      = 64;

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "create failed");
    TEST_CHECK(exsrc->get_effective_alignment() == 64,
               "the allocator must start with the configured alignment");
    TEST_CHECK(exsrc->get_l2_mem_kind() == llama_expert_mem_kind::PAGEABLE_HOST,
               "before any backend bind the arena must be PAGEABLE_HOST");

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(vk);
    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(dev_buft, 2 * expert_bytes);
    TEST_CHECK(bank_buf != nullptr, "failed to allocate the device bank");
    ggml_tensor bank = make_test_bank(bank_buf, 2 * expert_bytes);

    // El primer fill enlaza el backend y dispara la sustitucion transaccional.
    auto st = exsrc->fill_expert(vk, &bank, 0, &t, 0, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill through Vulkan failed");

    // La alineacion activa debe ser valida y coherente con la memoria real.
    const size_t eff = exsrc->get_effective_alignment();
    TEST_CHECK(eff != 0 && (eff & (eff - 1)) == 0 && eff % sizeof(void *) == 0,
               "the effective alignment must remain a valid power of two");
    TEST_CHECK(eff >= 64, "the effective alignment must never be smaller than the base");
    const auto kind = exsrc->get_l2_mem_kind();
    TEST_CHECK(kind == llama_expert_mem_kind::PINNED_VK ||
               kind == llama_expert_mem_kind::PAGEABLE_HOST_FALLBACK ||
               kind == llama_expert_mem_kind::PAGEABLE_HOST,
               "the memory classification must be honest");

    // El allocator sigue funcional: un segundo fill escribe en la arena activa.
    st = exsrc->fill_expert(vk, &bank, 0, &t, 1, expert_bytes, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "second fill on the substituted arena failed");
    TEST_CHECK(exsrc->get_arena_stats().blocks_live.load() == 2,
               "both blocks must live in the active arena");

    // Invariante del allocator intacto.
    const auto & s = exsrc->get_arena_stats();
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "arena invariant violated after substitution");

    // Teardown limpio: no doble liberacion.
    exsrc->drain();
    TEST_CHECK(!exsrc->was_arena_quarantined(), "drain after substitution must free the arena");
    exsrc.reset();

    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(vk);
    close(fd);
    unlink(temp_path);
    printf("[PASS] Blocker C arena alignment substitution passed.\n");
}

static void test_callback_sync_vulkan() {
    printf("[TEST] Testing CALLBACK_SYNC with a real Vulkan backend...\n");

    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (gpu_dev == nullptr) {
        printf("[SKIP] no GPU/iGPU backend: CALLBACK_SYNC not exercised in this build\n");
        return;
    }

    ggml_backend_t vk = ggml_backend_dev_init(gpu_dev, nullptr);
    TEST_CHECK(vk != nullptr, "failed to init the GPU backend");
    const char * vk_name = ggml_backend_name(vk);
    if (vk_name == nullptr || strstr(vk_name, "Vulkan") == nullptr) {
        printf("[SKIP] the GPU backend is not Vulkan (%s): CALLBACK_SYNC not exercised\n",
               vk_name ? vk_name : "unknown");
        ggml_backend_free(vk);
        return;
    }

    const size_t expert_bytes = 1024 * 1024;
    const size_t file_size    = 4 * expert_bytes;

    char temp_path[] = "/tmp/test_nexus_vk_XXXXXX";
    int fd = mkstemp(temp_path);
    TEST_CHECK(fd >= 0, "mkstemp failed");
    std::vector<uint8_t> pattern(expert_bytes);
    for (size_t i = 0; i < expert_bytes; ++i) {
        pattern[i] = (uint8_t) (i % 251);
    }
    TEST_CHECK(pwrite(fd, pattern.data(), expert_bytes, 0) == (ssize_t) expert_bytes, "pwrite failed");
    TEST_CHECK(ftruncate(fd, file_size) == 0, "ftruncate failed");

    llama_expert_model_meta meta;
    meta.files.push_back({ temp_path, file_size });
    ggml_tensor t{};
    t.ne[2] = 4; t.nb[2] = expert_bytes;
    meta.tensors[&t] = { 0, 0, expert_bytes, expert_bytes, 4, true };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 4 * expert_bytes;
    cfg.alignment      = 64;

    auto exsrc = llama_expert_source::create(meta, cfg).source;
    TEST_CHECK(exsrc != nullptr, "create (Vulkan) failed");

    // el banco es memoria de dispositivo, como en el MoE cache real
    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(vk);
    ggml_backend_buffer_t bank_buf = ggml_backend_buft_alloc_buffer(dev_buft, 2 * expert_bytes);
    TEST_CHECK(bank_buf != nullptr, "failed to allocate the device bank");
    ggml_tensor bank = make_test_bank(bank_buf, 2 * expert_bytes);

    auto st = exsrc->fill_expert(vk, &bank, 0, &t, 0, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "fill through the Vulkan backend failed");

    const auto kind = exsrc->get_l2_mem_kind();
    printf("  L2 memory kind: %s\n",
           kind == llama_expert_mem_kind::PINNED_VK ? "PINNED_VK" :
           (kind == llama_expert_mem_kind::PAGEABLE_HOST_FALLBACK ? "PAGEABLE_HOST_FALLBACK" : "PAGEABLE_HOST"));
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 1,
               "the Vulkan transfer must be inflight until confirmed");

    exsrc->begin_sync();
    ggml_backend_synchronize(vk);
    exsrc->end_sync();
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 0,
               "end_sync must confirm the Vulkan transfer");
    TEST_CHECK(exsrc->get_arena_stats().sync_calls.load() == 1,
               "end_sync must count exactly one synchronization");

    // la transferencia se emitio por el backend Vulkan y, tras sincronizar, los
    // bytes del experto estan en el banco
    std::vector<uint8_t> out(expert_bytes);
    ggml_backend_tensor_get(&bank, out.data(), 0, expert_bytes);
    TEST_CHECK(memcmp(out.data(), pattern.data(), expert_bytes) == 0,
               "the expert bytes did not reach the bank after the Vulkan transfer");

    // transferencia pendiente al iniciar el teardown: drain la confirma
    st = exsrc->fill_expert(vk, &bank, 0, &t, 1, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::LOADED, "second Vulkan fill failed");
    exsrc->drain();
    TEST_CHECK(!exsrc->was_arena_quarantined(),
               "drain must free the arena after confirming the Vulkan transfer");
    TEST_CHECK(exsrc->get_arena_stats().blocks_inflight.load() == 0,
               "drain must confirm the pending Vulkan transfer");
    st = exsrc->fill_expert(vk, &bank, 0, &t, 2, 0, expert_bytes);
    TEST_CHECK(st == llama_expert_fill_status::IO_ERROR, "fill after drain must be rejected");

    exsrc.reset();
    ggml_backend_buffer_free(bank_buf);
    ggml_backend_free(vk);

    close(fd);
    unlink(temp_path);
    printf("[PASS] CALLBACK_SYNC with the Vulkan backend passed.\n");
}

struct sched_abort_state {
    ggml_backend_sched_t sched = nullptr;
    bool invoked = false;
};

static bool sched_abort_copy_cb(ggml_backend_t /*backend*/, const ggml_tensor * /*src*/, ggml_tensor * /*dst*/, ggml_cgraph * /*graph*/, void * user_data) {
    auto * st = static_cast<sched_abort_state *>(user_data);
    st->invoked = true;
    // report the input as handled (true) and ask the scheduler to abort: returning
    // false here would silently perform the conventional copy
    ggml_backend_sched_abort(st->sched);
    return true;
}

static void test_sched_abort_from_copy_callback() {
    printf("[TEST] Testing scheduler abort requested by a copy callback...\n");

    // llama_backend_init() normally loads the backends; this test does not
    // initialize llama, so load them here to see the GPU devices
    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (gpu_dev == nullptr) {
        // the callback is only invoked for a split whose backend cannot read the
        // host weight directly; a CPU-only build has no such split
        printf("[SKIP] no GPU backend: copy-callback abort not exercised in this build\n");
        return;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    TEST_CHECK(backend_gpu != nullptr && backend_cpu != nullptr, "failed to initialize backends");

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_init_params ip = { 1024*1024, nullptr, true };

    // weights in a host buffer with WEIGHTS usage
    ggml_context * ctx_w = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 8, 8);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, cpu_buft);
    TEST_CHECK(buf_w != nullptr, "failed to allocate weight buffer");
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // input in a host buffer
    ggml_context * ctx_x = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, 8, 64);
    ggml_set_name(x, "x");
    ggml_backend_buffer_t buf_x = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, cpu_buft);
    TEST_CHECK(buf_x != nullptr, "failed to allocate input buffer");

    // graph: mul_mat on the GPU, with the host weight copied by the scheduler
    ggml_context * ctx_g = ggml_init(ip);
    ggml_tensor * y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backends[2] = { backend_gpu, backend_cpu };
    ggml_backend_buffer_type_t bufts[2] = {
        ggml_backend_get_default_buffer_type(backend_gpu),
        ggml_backend_get_default_buffer_type(backend_cpu),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, false, true);
    TEST_CHECK(sched != nullptr, "failed to create scheduler");

    sched_abort_state state;
    state.sched = sched;
    ggml_backend_sched_set_copy_callback(sched, sched_abort_copy_cb, &state);

    const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    TEST_CHECK(state.invoked, "the copy callback was not invoked");
    TEST_CHECK(status == GGML_STATUS_ABORTED, "the scheduler did not return GGML_STATUS_ABORTED");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_x);
    ggml_free(ctx_g);
    ggml_free(ctx_x);
    ggml_free(ctx_w);
    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);

    printf("[PASS] Scheduler abort from copy callback passed.\n");
}

int main() {
    printf("=== RUNNING NEXUS EXPERT SOURCE UNIT TESTS ===\n");
    test_pread_full();
    test_l2_dimensioning_and_pinning();
    test_allocator_coalescing_and_fragmentation();
    test_concurrency_and_barrier_quiescence();
    test_blocker_a_concurrent_same_key();
    test_blocker_b_create_statuses();
    test_blocker_c_alignment_transaction();
    test_d16_alloc_failure_injection();
    test_b1_b4_robustness_invariants();
    test_two_phase_sync_and_teardown();
    test_b2_sync_modes_and_teardown();
    test_b3_metadata_validation();
    test_b4_tensor_identity();
    test_b5_statistics();
    test_callback_sync_vulkan();
    test_sched_abort_from_copy_callback();
    printf("=== ALL NEXUS EXPERT SOURCE TESTS PASSED ===\n");
    return 0;
}
