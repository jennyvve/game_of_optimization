/**
 * Optimized Game of Life generation.
 * Author: Jenny Vermeltfoort, jennyvermeltfoort@outlook.com
 * Date: 1-8-2025
 */

#include <immintrin.h>

#include "stdint.h"
#include "stdio.h"
#include "time.h"

#define SIZE_WORLD_X_CONF \
    64  // actual size = val * sizeof(segment_t) * 8
#define SIZE_WORLD_Y_CONF 64  // actual size = val * GEN_CHUNK_SIZE

#define SCREEN_SIZE_X 4
#define SCREEN_SIZE_Y 48
#define SCREEN_SIZE (SCREEN_SIZE_X * SCREEN_SIZE_Y)

#define SIZE_BIT_MAP 9
#define GEN_CHUNK_SIZE 8
#define SIZE_SEGMENT (sizeof(segment_t) * 8)
#define SIZE_WORLD_X (SIZE_WORLD_X_CONF * SIZE_SEGMENT)
#define SIZE_WORLD_Y (SIZE_WORLD_Y_CONF * GEN_CHUNK_SIZE)
#define SIZE_WORLD (SIZE_WORLD_X_CONF * SIZE_WORLD_Y)

#define SEGMENT_FORMAT_SIZE 8

typedef uint32_t segment_t;
typedef segment_t segment_chunk_t[8];

typedef enum {
    cell_segment_read = 0,
    cell_segment_write = 1,
} cell_segment_rw_t;

typedef enum {
    cell_value_dead = 0,
    cell_value_alive = 1,
} cell_value_t;

typedef union {
    __m256i flat;
    uint32_t index[8];
} u256i_t;

const cell_value_t alive_cell_rules[] = {
    // index is amount of neighbours.
    [0] = cell_value_dead,  [1] = cell_value_dead,
    [2] = cell_value_alive, [3] = cell_value_alive,
    [4] = cell_value_dead,  [5] = cell_value_dead,
    [6] = cell_value_dead,  [7] = cell_value_dead,
    [8] = cell_value_dead,
};
const cell_value_t dead_cell_rules[] = {
    // index is amount of neighbours.
    [0] = cell_value_dead, [1] = cell_value_dead,
    [2] = cell_value_dead, [3] = cell_value_alive,
    [4] = cell_value_dead, [5] = cell_value_dead,
    [6] = cell_value_dead, [7] = cell_value_dead,
    [8] = cell_value_dead,
};
const cell_value_t* const cell_rules[] = {
    // index is the cell value.
    [cell_value_dead] = dead_cell_rules,
    [cell_value_alive] = alive_cell_rules,
};

segment_t buffer_1[SIZE_WORLD +
                   2];  // n + 1 read stub, n + 2 write stub. ZII
segment_t buffer_2[SIZE_WORLD +
                   2];  // n + 1 read stub, n + 2 write stub. ZII
segment_t* world = buffer_1;
segment_t* world_buffer = buffer_2;
char newline_table[SIZE_WORLD];
char segment_format[1 << SEGMENT_FORMAT_SIZE][SEGMENT_FORMAT_SIZE];
cell_value_t bitmap_translator[1 << 9];

// Calculate the segment index from x, y coordinates. When out of
// bounds return a read or a write stub, located at n + 1 or n + 2
// respectively.
inline uint32_t get_index(const cell_segment_rw_t rw,
                          const uint32_t x, const uint32_t y) {
    uint8_t oob = (y >= SIZE_WORLD_Y) || (x >= SIZE_WORLD_X_CONF);
    return (SIZE_WORLD_X_CONF * y + x) * !oob + SIZE_WORLD * oob +
           rw * oob;
}

inline segment_t get_cell_segment(const segment_t* const buf,
                                  const uint32_t x,
                                  const uint32_t y) {
    return buf[get_index(cell_segment_read, x, y)];
}

inline void set_cell_segment(segment_t* const buf,
                             const segment_t segment,
                             const uint32_t x, const uint32_t y) {
    buf[get_index(cell_segment_write, x, y)] = segment;
}

inline void set_cell_chunk(segment_t* const buf,
                           const segment_chunk_t chunk,
                           const uint32_t x, const uint32_t y) {
    buf[get_index(cell_segment_write, x, y + 0)] = chunk[0];
    buf[get_index(cell_segment_write, x, y + 1)] = chunk[1];
    buf[get_index(cell_segment_write, x, y + 2)] = chunk[2];
    buf[get_index(cell_segment_write, x, y + 3)] = chunk[3];
    buf[get_index(cell_segment_write, x, y + 4)] = chunk[4];
    buf[get_index(cell_segment_write, x, y + 5)] = chunk[5];
    buf[get_index(cell_segment_write, x, y + 6)] = chunk[6];
    buf[get_index(cell_segment_write, x, y + 7)] = chunk[7];
}

void set_cell_value(segment_t* const buf, const cell_value_t value,
                    const uint32_t x, const uint32_t y) {
    const uint8_t shift = (x % SIZE_SEGMENT);
    segment_t segment = get_cell_segment(buf, x / SIZE_SEGMENT, y);
    segment &= (~0 - (1 << shift));
    segment |= (value << shift);
    set_cell_segment(buf, segment, x / SIZE_SEGMENT, y);
}

void toggle_cell_value(segment_t* const buf, const uint32_t x,
                       const uint32_t y) {
    const uint8_t shift = (x % SIZE_SEGMENT);
    segment_t segment = get_cell_segment(buf, x / SIZE_SEGMENT, y);
    segment ^= (1 << shift);
    set_cell_segment(buf, segment, x / SIZE_SEGMENT, y);
}

inline uint8_t cpy_segment_format(char* buf, char segment_format[8]) {
    for (uint8_t i = 0; i < SEGMENT_FORMAT_SIZE; i++) {
        buf[i] = segment_format[i];
    }
    return SEGMENT_FORMAT_SIZE;
}

void print_world(uint32_t x, uint32_t y) {
    char buf[SCREEN_SIZE * (SIZE_SEGMENT + 1) + 7] = {
        "\033[1;1H"};  // newline char, and cursor pos (6 bytes).
    char* ptr = buf + 6;

    for (uint16_t i = 0; i < SCREEN_SIZE; i++) {
        uint32_t xpos = x + i % SCREEN_SIZE_X;
        uint32_t ypos = y + i / SCREEN_SIZE_X;
        segment_t segment = world[ypos * SIZE_WORLD_X_CONF + xpos];
        *ptr++ = newline_table[i];
        ptr += cpy_segment_format(
            ptr, segment_format[(segment >> 0) & 0XFF]);
        ptr += cpy_segment_format(
            ptr, segment_format[(segment >> 8) & 0XFF]);
        ptr += cpy_segment_format(
            ptr, segment_format[(segment >> 16) & 0XFF]);
        ptr += cpy_segment_format(
            ptr, segment_format[(segment >> 24) & 0XFF]);
    }

    fwrite(buf, 1, ptr - buf, stdout);
}

void generate_segment_format(const char alive, const char dead) {
    for (uint32_t i = 0; i < (1 << SEGMENT_FORMAT_SIZE); i++) {
        segment_format[i][0] = (i & (1 << 0)) ? alive : dead;
        segment_format[i][1] = (i & (1 << 1)) ? alive : dead;
        segment_format[i][2] = (i & (1 << 2)) ? alive : dead;
        segment_format[i][3] = (i & (1 << 3)) ? alive : dead;
        segment_format[i][4] = (i & (1 << 4)) ? alive : dead;
        segment_format[i][5] = (i & (1 << 5)) ? alive : dead;
        segment_format[i][6] = (i & (1 << 6)) ? alive : dead;
        segment_format[i][7] = (i & (1 << 7)) ? alive : dead;
    }
}

void generate_newline_table(void) {
    for (uint16_t i = 1; i < SCREEN_SIZE; i++) {
        newline_table[i] = (i % SCREEN_SIZE_X) ? '\0' : '\n';
    }
}

void generate_bitmap_translation_table(void) {
    for (uint16_t i = 0; i < (1 << 9); i++) {
        uint8_t neighbours = (i >> 0 & 0X1) + (i >> 1 & 0X1) +
                             (i >> 2 & 0X1) + (i >> 3 & 0X1) +
                             (i >> 5 & 0X1) + (i >> 6 & 0X1) +
                             (i >> 7 & 0X1) + (i >> 8 & 0X1);
        uint8_t cell_value = (i >> 4 & 0X1);
        bitmap_translator[i] = cell_rules[cell_value][neighbours];
    }
}

inline __m256i avx_msk_lsh(__m256i val, const uint32_t mask,
                           const uint32_t lsh) {
    __m256i acc = _mm256_and_si256(val, _mm256_set1_epi32(mask));
    return _mm256_slli_epi32(acc, lsh);
}

inline __m256i avx_rsh_msk_lsh(__m256i val, const uint32_t rsh,
                               const uint32_t mask,
                               const uint32_t lsh) {
    __m256i acc = _mm256_srli_epi32(val, rsh);
    acc = _mm256_and_si256(acc, _mm256_set1_epi32(mask));
    return _mm256_slli_epi32(acc, lsh);
}

inline __m256i avx_calc_segments(__m256i top, __m256i mid,
                                 __m256i bot) {
    __m256i acc = avx_msk_lsh(top, 0X7, 0);
    __m256i acc_r = _mm256_add_epi32(_mm256_setzero_si256(), acc);
    acc = avx_msk_lsh(mid, 0X7, 3);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(bot, 0X7, 6);
    acc_r = _mm256_add_epi32(acc_r, acc);
    return acc_r;
}

inline __m256i avx_calc_right_edge_segments(__m256i top, __m256i mid,
                                            __m256i bot,
                                            __m256i right_top,
                                            __m256i right_mid,
                                            __m256i right_bot) {
    __m256i acc = avx_msk_lsh(top, 0X3, 0);
    __m256i acc_r = _mm256_add_epi32(_mm256_setzero_si256(), acc);
    acc = avx_msk_lsh(right_top, 0X1, 2);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(mid, 0X3, 3);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(right_mid, 0X1, 5);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(bot, 0X3, 6);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(right_bot, 0X1, 8);
    acc_r = _mm256_add_epi32(acc_r, acc);
    return acc_r;
}

inline __m256i avx_calc_left_edge_segments(__m256i top, __m256i mid,
                                           __m256i bot,
                                           __m256i left_top,
                                           __m256i left_mid,
                                           __m256i left_bot) {
    __m256i acc = avx_msk_lsh(top, 0X3, 1);
    __m256i acc_r = _mm256_add_epi32(_mm256_setzero_si256(), acc);
    acc = avx_msk_lsh(mid, 0X3, 4);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_msk_lsh(bot, 0X3, 7);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_rsh_msk_lsh(left_top, (SIZE_SEGMENT - 1), 0X1, 0);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_rsh_msk_lsh(left_mid, (SIZE_SEGMENT - 1), 0X1, 3);
    acc_r = _mm256_add_epi32(acc_r, acc);
    acc = avx_rsh_msk_lsh(left_bot, (SIZE_SEGMENT - 1), 0X1, 6);
    acc_r = _mm256_add_epi32(acc_r, acc);
    return acc_r;
}

void fill_segment_chunks(segment_t* buf, uint32_t x, uint32_t y,
                         segment_chunk_t top, segment_chunk_t mid,
                         segment_chunk_t bot) {
    mid[0] = get_cell_segment(buf, x, y + 0);
    mid[1] = get_cell_segment(buf, x, y + 1);
    mid[2] = get_cell_segment(buf, x, y + 2);
    mid[3] = get_cell_segment(buf, x, y + 3);
    mid[4] = get_cell_segment(buf, x, y + 4);
    mid[5] = get_cell_segment(buf, x, y + 5);
    mid[6] = get_cell_segment(buf, x, y + 6);
    mid[7] = get_cell_segment(buf, x, y + 7);

    top[0] = get_cell_segment(buf, x, y - 1);
    top[1] = mid[0];
    top[2] = mid[1];
    top[3] = mid[2];
    top[4] = mid[3];
    top[5] = mid[4];
    top[6] = mid[5];
    top[7] = mid[6];

    bot[0] = mid[1];
    bot[1] = mid[2];
    bot[2] = mid[3];
    bot[3] = mid[4];
    bot[4] = mid[5];
    bot[5] = mid[6];
    bot[6] = mid[7];
    bot[7] = get_cell_segment(buf, x, y + 8);
}

// Generates a bitmap for each bit in a segment, then translates this
// bitmap into the cell value. Calculate 8 chunks at a time in AVX2.
// Bitmap represents the cell and its neighbours, 0 .. 8 neighbours,
// with 4 the cell:
// - 0 1 2
// - 3 4 5
// - 6 7 8
void generate_segment_chunks(segment_t* buf, segment_chunk_t chunk,
                             uint32_t x, uint32_t y) {
    u256i_t bitmaps[SIZE_SEGMENT] = {};

    segment_chunk_t top;
    segment_chunk_t mid = {};
    segment_chunk_t bot;
    fill_segment_chunks(buf, x, y, top, mid, bot);

    segment_chunk_t left_top;
    segment_chunk_t left_mid;
    segment_chunk_t left_bot;
    fill_segment_chunks(buf, x - 1, y, left_top, left_mid, left_bot);

    segment_chunk_t right_top;
    segment_chunk_t right_mid;
    segment_chunk_t right_bot;
    fill_segment_chunks(buf, x + 1, y, right_top, right_mid,
                        right_bot);

    __m256i lst = _mm256_load_si256((__m256i*)left_top);
    __m256i st = _mm256_load_si256((__m256i*)top);
    __m256i rst = _mm256_load_si256((__m256i*)right_top);

    __m256i lsm = _mm256_load_si256((__m256i*)left_mid);
    __m256i sm = _mm256_load_si256((__m256i*)mid);
    __m256i rsm = _mm256_load_si256((__m256i*)right_mid);

    __m256i lsb = _mm256_load_si256((__m256i*)left_bot);
    __m256i sb = _mm256_load_si256((__m256i*)bot);
    __m256i rsb = _mm256_load_si256((__m256i*)right_bot);

    // order matters, calculate left edge then mid then right edge.
    bitmaps[0].flat =
        avx_calc_left_edge_segments(st, sm, sb, lst, lsm, lsb);

    for (uint8_t i = 1; i < (SIZE_SEGMENT - 1); i++) {
        bitmaps[i].flat = avx_calc_segments(st, sm, sb);
        st = _mm256_srli_epi32(st, 1);
        sm = _mm256_srli_epi32(sm, 1);
        sb = _mm256_srli_epi32(sb, 1);
    }

    bitmaps[SIZE_SEGMENT - 1].flat =
        avx_calc_right_edge_segments(st, sm, sb, rst, rsm, rsb);

    for (uint8_t i = 0; i < SIZE_SEGMENT; i++) {
        chunk[0] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[0]] << i);
        chunk[1] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[1]] << i);
        chunk[2] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[2]] << i);
        chunk[3] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[3]] << i);
        chunk[4] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[4]] << i);
        chunk[5] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[5]] << i);
        chunk[6] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[6]] << i);
        chunk[7] +=
            ((uint32_t)bitmap_translator[bitmaps[i].index[7]] << i);
    }
}

void generate_iteration(void) {
    segment_t* buf = world;
    world = world_buffer;
    world_buffer = buf;

    for (uint32_t i = 0; i < SIZE_WORLD_Y_CONF * SIZE_WORLD_X_CONF;
         i++) {
        uint32_t xpos = i % SIZE_WORLD_X_CONF;
        uint32_t ypos = (i / SIZE_WORLD_X_CONF) * GEN_CHUNK_SIZE;
        segment_chunk_t chunk = {};
        generate_segment_chunks(world_buffer, chunk, xpos, ypos);
        set_cell_chunk(world, chunk, xpos, ypos);
    }
}

void infest_random(void) {
    srand(time(NULL));
    for (uint32_t i = 0; i < SIZE_WORLD * GEN_CHUNK_SIZE; i++) {
        toggle_cell_value(world, rand() % SIZE_WORLD_X,
                          rand() % SIZE_WORLD_Y);
    }
}

int main(void) {
    generate_segment_format('x', '-');
    generate_newline_table();
    generate_bitmap_translation_table();

    // infest_random();

    // Places kok's galaxy inside the world.
    set_cell_value(world, cell_value_alive, 25 + 5, 5);
    set_cell_value(world, cell_value_alive, 25 + 6, 5);
    set_cell_value(world, cell_value_alive, 25 + 7, 5);
    set_cell_value(world, cell_value_alive, 25 + 8, 5);
    set_cell_value(world, cell_value_alive, 25 + 9, 5);
    set_cell_value(world, cell_value_alive, 25 + 10, 5);
    set_cell_value(world, cell_value_alive, 25 + 5, 6);
    set_cell_value(world, cell_value_alive, 25 + 6, 6);
    set_cell_value(world, cell_value_alive, 25 + 7, 6);
    set_cell_value(world, cell_value_alive, 25 + 8, 6);
    set_cell_value(world, cell_value_alive, 25 + 9, 6);
    set_cell_value(world, cell_value_alive, 25 + 10, 6);
    set_cell_value(world, cell_value_alive, 25 + 8, 12);
    set_cell_value(world, cell_value_alive, 25 + 9, 12);
    set_cell_value(world, cell_value_alive, 25 + 10, 12);
    set_cell_value(world, cell_value_alive, 25 + 11, 12);
    set_cell_value(world, cell_value_alive, 25 + 12, 12);
    set_cell_value(world, cell_value_alive, 25 + 13, 12);
    set_cell_value(world, cell_value_alive, 25 + 8, 13);
    set_cell_value(world, cell_value_alive, 25 + 9, 13);
    set_cell_value(world, cell_value_alive, 25 + 10, 13);
    set_cell_value(world, cell_value_alive, 25 + 11, 13);
    set_cell_value(world, cell_value_alive, 25 + 12, 13);
    set_cell_value(world, cell_value_alive, 25 + 13, 13);
    set_cell_value(world, cell_value_alive, 25 + 12, 5);
    set_cell_value(world, cell_value_alive, 25 + 13, 5);
    set_cell_value(world, cell_value_alive, 25 + 12, 6);
    set_cell_value(world, cell_value_alive, 25 + 13, 6);
    set_cell_value(world, cell_value_alive, 25 + 12, 7);
    set_cell_value(world, cell_value_alive, 25 + 13, 7);
    set_cell_value(world, cell_value_alive, 25 + 12, 8);
    set_cell_value(world, cell_value_alive, 25 + 13, 8);
    set_cell_value(world, cell_value_alive, 25 + 12, 9);
    set_cell_value(world, cell_value_alive, 25 + 13, 9);
    set_cell_value(world, cell_value_alive, 25 + 12, 10);
    set_cell_value(world, cell_value_alive, 25 + 13, 10);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 5 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 5 + 3);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 6 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 6 + 3);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 7 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 7 + 3);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 8 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 8 + 3);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 9 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 9 + 3);
    set_cell_value(world, cell_value_alive, 25 + 12 - 7, 10 + 3);
    set_cell_value(world, cell_value_alive, 25 + 13 - 7, 10 + 3);

    printf("\033[2J");
    printf("\033[?25l");

    for (uint32_t i = 0; i < 10000; i++) {
        print_world(0, 0);
        generate_iteration();
    }

    printf("\n");
    printf("\033[?25h");

    return 0;
}
