// Single-game bitboard engine for the classic-mode rules (docs/RULES.md), a twin of game.py.
//
// Board: unsigned __int128, bit r * 10 + c is cell (row r, col c), row 0 = top; bits 100..127 stay 0.
// Pieces are ids into pieces.all_pieces(table); a move is (slot, pos) with pos = row * 10 + col of the
// piece's bounding-box top-left. Tables come from Python, so pieces.json stays the only piece source.
#pragma once

#include <algorithm>
#include <bit>
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

// ---- board operations -------------------------------------------------------------------------

struct State {
    Board board = 0;
    int pending[3] = {};  // piece ids; a move removes one slot and the spawned piece goes last
    int64_t score = 0, moves = 0, lines = 0;
    int history[3] = {-1, -1, -1};  // spawned groups, most recent first; persists across resets
};

struct Placed {
    Board board;  // after placement and line clears
    int points;
    int lines;
};

inline constexpr Board ROW_MASK = 0x3FF;

// Bitmask of the top-left positions where ``piece`` fits: q is blocked when some cell q + off is filled.
// valid[] only holds positions whose bounding box fits, so the shifts never wrap across a row.
inline Board legal_positions(const Tables& t, Board board, int piece) {
    Board blocked = 0;
    for (int k = 0; k < t.size[piece]; k++) blocked |= board >> t.offsets[piece][k];
    return t.valid[piece] & ~blocked;
}

// Place ``piece`` at ``pos`` (legality is not checked), then clear every full row and column together.
inline Placed apply(const Tables& t, Board board, int piece, int pos) {
    const Board b = board | t.shape[piece] << pos;
    Board cols = ROW_MASK;  // AND of the 10 row fields = full-column bits
    Board cleared = 0;
    int lines = 0;
    for (int r = 0; r < N; r++) {
        const Board row = b >> (N * r) & ROW_MASK;
        cols &= row;
        if (row == ROW_MASK) cleared |= ROW_MASK << (N * r), lines++;
    }
    if (cols) {
        lines += std::popcount(static_cast<uint64_t>(cols));
        for (int r = 0; r < N; r++) cleared |= cols << (N * r);
    }
    return {b & ~cleared, 10 * t.size[piece] + 50 * lines * (lines + 1), lines};
}

// Calls f(pos) for each set bit of ``mask``, ascending.
template <class F>
inline void for_each_bit(Board mask, F&& f) {
    for (auto lo = static_cast<uint64_t>(mask); lo; lo &= lo - 1) f(std::countr_zero(lo));
    for (auto hi = static_cast<uint64_t>(mask >> 64); hi; hi &= hi - 1) f(64 + std::countr_zero(hi));
}

inline bool has_legal_move(const Tables& t, const State& s) {
    for (int slot = 0; slot < 3; slot++)
        if (legal_positions(t, s.board, s.pending[slot])) return true;
    return false;
}

// restart(): empty board, score 0, three fresh pieces spawned at score 0. The spawn history is kept.
template <class Rng>
void reset(const Tables& t, State& s, Rng& rng) {
    s.board = 0;
    s.score = s.moves = s.lines = 0;
    for (int& piece : s.pending) piece = spawn(t, 0, s.history, rng);
}

// onTouchEnded (game.Game.step): check the move, remove the slot (the other two keep their order),
// spawn with the score *before* this move and append it, then place, clear and score.
template <class Rng>
Placed step(const Tables& t, State& s, int slot, int pos, Rng& rng) {
    if (slot < 0 || slot > 2 || pos < 0 || pos >= CELLS) throw std::invalid_argument("move out of range");
    const int piece = s.pending[slot];
    if (!(legal_positions(t, s.board, piece) >> pos & 1)) throw std::invalid_argument("illegal move");
    int k = 0;
    for (int i = 0; i < 3; i++)
        if (i != slot) s.pending[k++] = s.pending[i];
    s.pending[2] = spawn(t, s.score, s.history, rng);
    const Placed a = apply(t, s.board, piece, pos);
    s.board = a.board;
    s.score += a.points;
    s.moves++;
    s.lines += a.lines;
    return a;
}

}  // namespace bpa
