// Single-game bitboard engine for the classic-mode rules (docs/RULES.md), a twin of game.py.
//
// Board: unsigned __int128, bit r * 10 + c is cell (row r, col c), row 0 = top; bits 100..127 stay 0.
// Pieces are ids into pieces.all_pieces(table); a move is (slot, pos) with pos = row * 10 + col of the
// piece's bounding-box top-left. Tables come from Python, so pieces.json stays the only piece source.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bpa {

using Board = unsigned __int128;

inline constexpr int N = 10;
inline constexpr int CELLS = N * N;
inline constexpr int MAX_PIECES = 32;
inline constexpr int MAX_GROUPS = 16;
inline constexpr int MAX_CELLS = 9;
inline constexpr int COOLDOWN_GROUPS = 8;  // after a hard group, randTiles picks among groups 0..7
inline constexpr int64_t SPAWN_THRESHOLDS[3] = {2999, 5999, 11999};

struct Tables {
    int n_pieces = 0;
    int n_groups = 0;
    int size[MAX_PIECES] = {};  // cells per piece
    int group[MAX_PIECES] = {};
    int offsets[MAX_PIECES][MAX_CELLS] = {};  // r * 10 + c of each cell, relative to the top-left
    Board shape[MAX_PIECES] = {};  // the piece at position 0
    Board valid[MAX_PIECES] = {};  // top-left positions where the bounding box fits
    int group_offset[MAX_GROUPS] = {};  // first piece id of each group
    int group_size[MAX_GROUPS] = {};
    bool hard[MAX_GROUPS] = {};
};

inline Board bit(int i) { return Board{1} << i; }

inline Tables make_tables(
    const std::vector<std::vector<int>>& cells, const std::vector<int>& groups, const std::vector<int>& hard_groups) {
    auto fail = [](const char* what) { throw std::invalid_argument(what); };
    if (cells.empty() || cells.size() > MAX_PIECES) fail("Tables: 1..32 pieces");
    if (groups.size() != cells.size()) fail("Tables: one group per piece");
    Tables t;
    t.n_pieces = static_cast<int>(cells.size());
    for (int p = 0; p < t.n_pieces; p++) {
        const int g = groups[p];
        const int expected_next = p == 0 ? 0 : groups[p - 1] + 1;
        if (g != (p == 0 ? 0 : groups[p - 1]) && g != expected_next) fail("Tables: groups must be contiguous from 0");
        if (g >= MAX_GROUPS) fail("Tables: at most 16 groups");
        if (cells[p].empty() || cells[p].size() > MAX_CELLS) fail("Tables: 1..9 cells per piece");
        int height = 0, width = 0;
        for (size_t k = 0; k < cells[p].size(); k++) {
            const int off = cells[p][k];
            if (off < 0 || off >= CELLS) fail("Tables: cell offset outside the board");
            t.offsets[p][k] = off;
            t.shape[p] |= bit(off);
            height = std::max(height, off / N + 1);
            width = std::max(width, off % N + 1);
        }
        t.size[p] = static_cast<int>(cells[p].size());
        t.group[p] = g;
        for (int row = 0; row + height <= N; row++)
            for (int col = 0; col + width <= N; col++) t.valid[p] |= bit(row * N + col);
        if (g == t.n_groups) t.group_offset[t.n_groups++] = p;
        t.group_size[g]++;
    }
    for (int g : hard_groups) {
        if (g < 0 || g >= t.n_groups) fail("Tables: hard group out of range");
        t.hard[g] = true;
    }
    if (t.n_groups < COOLDOWN_GROUPS) fail("Tables: at least 8 groups (cooldown picks among 0..7)");
    return t;
}

// ---- spawning: PendingTileManager::randTiles (game.PieceGenerator.spawn) ----------------------

// v = (int)(r / 1000.0f * score), in float32 like the game
inline int64_t spawn_level(int64_t score, uint32_t r) {
    return static_cast<int64_t>(static_cast<float>(r) / 1000.0f * static_cast<float>(score));
}

inline int allowed_groups(const Tables& t, int64_t level) {
    int n = t.n_groups - 3;
    for (int64_t threshold : SPAWN_THRESHOLDS) n += level > threshold;
    return n;
}

inline bool in_cooldown(const Tables& t, const int history[3]) {
    for (int i = 0; i < 3; i++)
        if (history[i] >= 0 && t.hard[history[i]]) return true;
    return false;
}

// Draws exactly what PieceGenerator.spawn draws (the overridden group draw and randrange(1) included),
// pushes the group onto ``history`` (most recent first) and returns the piece id.
template <class Rng>
int spawn(const Tables& t, int64_t score, int history[3], Rng& rng) {
    const int64_t level = spawn_level(score, rng.randrange(1000));
    int group = static_cast<int>(rng.randrange(allowed_groups(t, level)));
    if (in_cooldown(t, history)) group = static_cast<int>(rng.randrange(COOLDOWN_GROUPS));
    history[2] = history[1];
    history[1] = history[0];
    history[0] = group;
    return t.group_offset[group] + static_cast<int>(rng.randrange(t.group_size[group]));
}

// Exact probability of each piece id being spawned next (batch_env.spawn_distribution); out[n_pieces].
inline void spawn_distribution(const Tables& t, int64_t score, const int history[3], double* out) {
    double p_group[MAX_GROUPS] = {};
    if (in_cooldown(t, history)) {
        for (int g = 0; g < COOLDOWN_GROUPS; g++) p_group[g] = 1.0 / COOLDOWN_GROUPS;
    } else {
        int count[MAX_GROUPS + 1] = {};  // how many of the 1000 values of r allow each group count
        for (uint32_t r = 0; r < 1000; r++) count[allowed_groups(t, spawn_level(score, r))]++;
        for (int allowed = 1; allowed <= t.n_groups; allowed++)
            for (int g = 0; g < allowed; g++) p_group[g] += count[allowed] / 1000.0 / allowed;
    }
    for (int p = 0; p < t.n_pieces; p++) out[p] = p_group[t.group[p]] / t.group_size[t.group[p]];
}

}  // namespace bpa
