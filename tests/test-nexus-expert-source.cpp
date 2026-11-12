#include "../src/llama-expert-source.h"
#include "../ggml/include/ggml-alloc.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST FAILURE at %s:%d: %s\n", __FILE__, __LINE__, msg); \
            abort(); \
        } \
    } while (0)

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

    meta.tensors[&t_gate_up] = { 0, 0, sz_gate_up, sz_gate_up, 128, true };
    meta.tensors[&t_down]    = { 0, sz_gate_up * 10, sz_down, sz_down, 128, true };

    llama_expert_source::config cfg;
    cfg.ram_warm_bytes = 1024ull * 1024 * 1024; // 1024 MiB

    auto exsrc = llama_expert_source::create(meta, cfg);
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

    auto exsrc = llama_expert_source::create(meta, cfg);
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

    auto exsrc = llama_expert_source::create(meta, cfg);
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

    auto exsrc = llama_expert_source::create(meta, cfg);
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
    // extents fuera del archivo: pread_full debe fallar y propagar IO_ERROR
    meta.tensors[&t_bad] = { 0, fake_file_size + 4096, 1000000, 1000000, 5, true };

    auto exsrc = llama_expert_source::create(meta, cfg);
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

    // Error de lectura: se propaga y el bloque no queda como RESIDENT valido
    st = exsrc->fill_expert(cpu, &bank, 0, &t_bad, 0, 0, 1000000);
    TEST_CHECK(st == llama_expert_fill_status::IO_ERROR, "pread failure must return IO_ERROR");

    const auto & s = exsrc->get_arena_stats();
    TEST_CHECK(s.aligned_bytes.load() + s.slack_bytes.load() + s.free_bytes.load() == s.arena_bytes.load(),
               "arena invariant violated after a failed read");

    // Destruccion con una transferencia aun pendiente: drain_for_teardown la
    // confirma antes de liberar la arena.
    exsrc.reset();

    // Destruccion sin backend enlazado (modo HOST_ONLY)
    auto exsrc_host = llama_expert_source::create(meta, cfg);
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
    test_b1_b4_robustness_invariants();
    test_two_phase_sync_and_teardown();
    test_sched_abort_from_copy_callback();
    printf("=== ALL NEXUS EXPERT SOURCE TESTS PASSED ===\n");
    return 0;
}
