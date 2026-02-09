// pt2bin: Convert silence_latent.pt -> .bin (raw float32, transposed)
// PyTorch .pt = ZIP with uncompressed data/0 entry containing raw floats
// Input shape: [1, 64, 15000] -> squeeze+transpose -> Output: [15000, 64]
// Zero dependencies beyond libc.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static constexpr int ROWS = 64;
static constexpr int COLS = 15000;
static constexpr size_t N_FLOATS = ROWS * COLS;
static constexpr size_t DATA_SIZE = N_FLOATS * sizeof(float);

// ZIP End of Central Directory signature
static constexpr uint32_t EOCD_SIG = 0x06054b50;
// ZIP Central Directory File Header signature
static constexpr uint32_t CDFH_SIG = 0x02014b50;
// ZIP Local File Header signature
static constexpr uint32_t LFH_SIG  = 0x04034b50;

template<typename T>
static T read_le(const uint8_t* p) {
    T v = 0;
    for (size_t i = 0; i < sizeof(T); i++)
        v |= static_cast<T>(p[i]) << (8 * i);
    return v;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: pt2bin <input.pt> <output.bin>\n");
        return 1;
    }

    // Read entire .pt file
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    auto* buf = static_cast<uint8_t*>(malloc(fsize));
    if (fread(buf, 1, fsize, f) != fsize) { perror("fread"); return 1; }
    fclose(f);

    // Find EOCD (last 22+ bytes of file)
    size_t eocd_pos = 0;
    for (size_t i = fsize - 22; i > 0; i--) {
        if (read_le<uint32_t>(buf + i) == EOCD_SIG) { eocd_pos = i; break; }
    }
    if (!eocd_pos) { fprintf(stderr, "Not a ZIP file\n"); return 1; }

    uint32_t cd_offset = read_le<uint32_t>(buf + eocd_pos + 16);
    uint16_t cd_entries = read_le<uint16_t>(buf + eocd_pos + 10);

    // Walk central directory, find entry ending with "/data/0"
    uint32_t local_offset = 0;
    uint32_t uncomp_size = 0;
    bool found = false;
    size_t pos = cd_offset;
    for (int i = 0; i < cd_entries && pos < fsize; i++) {
        if (read_le<uint32_t>(buf + pos) != CDFH_SIG) break;
        uint32_t cd_uncomp  = read_le<uint32_t>(buf + pos + 24);
        uint16_t name_len  = read_le<uint16_t>(buf + pos + 28);
        uint16_t extra_len = read_le<uint16_t>(buf + pos + 30);
        uint16_t comm_len  = read_le<uint16_t>(buf + pos + 32);
        uint32_t lhdr_off  = read_le<uint32_t>(buf + pos + 42);
        const char* name = reinterpret_cast<const char*>(buf + pos + 46);

        if (name_len >= 7 && memcmp(name + name_len - 7, "/data/0", 7) == 0) {
            local_offset = lhdr_off;
            uncomp_size = cd_uncomp;
            found = true;
            break;
        }
        pos += 46 + name_len + extra_len + comm_len;
    }
    if (!found) { fprintf(stderr, "Entry 'data/0' not found in ZIP\n"); return 1; }

    // Parse local file header to get data offset
    if (read_le<uint32_t>(buf + local_offset) != LFH_SIG) {
        fprintf(stderr, "Bad local file header\n"); return 1;
    }
    uint16_t lh_name_len  = read_le<uint16_t>(buf + local_offset + 26);
    uint16_t lh_extra_len = read_le<uint16_t>(buf + local_offset + 28);
    size_t data_offset = local_offset + 30 + lh_name_len + lh_extra_len;

    if (uncomp_size != DATA_SIZE) {
        fprintf(stderr, "Unexpected data size: %u (expected %zu)\n", uncomp_size, DATA_SIZE);
        return 1;
    }

    // Transpose [64, 15000] -> [15000, 64] and write
    const float* src = reinterpret_cast<const float*>(buf + data_offset);
    auto* dst = static_cast<float*>(malloc(DATA_SIZE));

    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            dst[j * ROWS + i] = src[i * COLS + j];

    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 1; }
    fwrite(dst, sizeof(float), N_FLOATS, out);
    fclose(out);

    printf("[Convert] [%d, %d] float32 -> %zu bytes\n", COLS, ROWS, DATA_SIZE);

    free(buf);
    free(dst);
    return 0;
}
