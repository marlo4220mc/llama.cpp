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
    cfg.ring_bytes     = 64ull * 1024 * 1024;   // 64 MiB

    auto exsrc = llama_expert_source::create(meta, cfg);
    TEST_CHECK(exsrc != nullptr, "Failed to create llama_expert_source");

    // Condición 1 del usuario:
    // La capacidad DEBE ser de exactamente 320 pares completos (no 321)
    size_t l2_pairs = exsrc->get_l2_capacity_pairs();
    printf("  L2 capacity in pairs: %zu (Expected: 320)\n", l2_pairs);
    TEST_CHECK(l2_pairs == 320, "L2 capacity must be exactly 320 pairs");

    // Staging ring slots para 64 MiB con slot de 2,230,272 bytes:
    // 67,108,864 / 2,230,272 = 30 slots
    size_t ring_slots = exsrc->get_staging_ring_slots();
    printf("  Staging ring slots: %zu (Expected: 30)\n", ring_slots);
    TEST_CHECK(ring_slots == 30, "Staging ring must have 30 slots");

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

int main() {
    printf("=== RUNNING NEXUS EXPERT SOURCE UNIT TESTS ===\n");
    test_pread_full();
    test_l2_dimensioning_and_pinning();
    printf("=== ALL NEXUS EXPERT SOURCE TESTS PASSED ===\n");
    return 0;
}
