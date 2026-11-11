#include "../src/llama-expert-source.h"
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

int main() {
    printf("=== RUNNING NEXUS EXPERT SOURCE UNIT TESTS ===\n");
    test_pread_full();
    test_l2_dimensioning_and_pinning();
    test_allocator_coalescing_and_fragmentation();
    test_concurrency_and_barrier_quiescence();
    printf("=== ALL NEXUS EXPERT SOURCE TESTS PASSED ===\n");
    return 0;
}
