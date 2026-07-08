/**
 * lkmc-wasm.cpp  —  Hexagonal Lattice KMC Electrodeposition Simulator (WASM port)
 *
 * C++ port of HexagonalLKMC_v3.py (a.k.a. LKMC_v5_gui_fast.py), matching the
 * *hexagonal* 6-neighbor lattice with row-parity offsets, the DEPOSITED /
 * PASSIVATED conditional bonding rule, and the passivation event kind.
 *
 * NOTE: this replaces an earlier lkmc-wasm.cpp that was a port of an older,
 * *square* 4-neighbor lattice version (LKMC_v2_commented_b.py) and did not
 * have PASSIVATED or passivation events at all. If you diff against that
 * file, expect it to look substantially different — that's expected.
 *
 * Build (native, for testing on your own machine):
 *   g++ -O2 -std=c++17 -o lkmc lkmc-wasm.cpp
 *   ./lkmc
 *
 * Build (WASM, requires the Emscripten SDK — see https://emscripten.org/docs/index.html):
 *   emcc lkmc-wasm.cpp -O3 --bind -sMODULARIZE -sEXPORT_ES6 -sALLOW_MEMORY_GROWTH -o lkmc-wasm.js
 *
 * JS/React usage sketch
 * ----------------------
 *   const mod = await createLkmcModule();
 *   mod._init_simulation(Nx, Ny, T, d0, e0, e1, nu_f, nu_d, nu_p, periodic_x);
 *   // optional: mod._set_pcg_state(...8 uint32 halves from get_pcg64_state.py...);
 *   function tick() {
 *     mod._run_steps(1000);                       // advance 1000 KMC events
 *     const ptr  = mod._get_lattice_ptr();
 *     const size = mod._get_lattice_size();
 *     const grid = mod.HEAPU8.subarray(ptr, ptr + size); // read-only view, no copy
 *     setSimState({ step: mod._get_step(), time: mod._get_time(), grid: grid.slice() });
 *     requestAnimationFrame(tick);
 *   }
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifndef __EMSCRIPTEN__
#include <filesystem>
#endif
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#ifndef __EMSCRIPTEN__
namespace fs = std::filesystem;
#endif

// ---------------------------------------------------------------------------
// PCG64 — identical to numpy.random.default_rng() draw sequence.
// (Unchanged from the original lkmc-wasm.cpp — already verified correct.)
// ---------------------------------------------------------------------------
struct PCG64State {
    uint64_t state_hi = 0x50c3ed493ae78588ULL;  // default: numpy seed=394583
    uint64_t state_lo = 0x2c8bef01c72f99e5ULL;
    uint64_t inc_hi   = 0x71a5befeec2f5ccaULL;
    uint64_t inc_lo   = 0x4df2b37d5d7aa1cbULL;
};

class PCG64 {
public:
    explicit PCG64(const PCG64State& s)
        : s_hi_(s.state_hi), s_lo_(s.state_lo),
          i_hi_(s.inc_hi),   i_lo_(s.inc_lo) {}

    double next_double() {
        advance();
        return (double)(xsl_rr() >> 11u) * (1.0 / 9007199254740992.0);
    }

    void reset(const PCG64State& s) {
        s_hi_ = s.state_hi; s_lo_ = s.state_lo;
        i_hi_ = s.inc_hi;   i_lo_ = s.inc_lo;
    }

private:
    uint64_t s_hi_, s_lo_, i_hi_, i_lo_;

    void advance() {
        __uint128_t s   = ((__uint128_t)s_hi_ << 64) | s_lo_;
        __uint128_t inc = ((__uint128_t)i_hi_ << 64) | i_lo_;
        const __uint128_t MUL =
            ((__uint128_t)0x2360ed051fc65da4ULL << 64) | 0x4385df649fccf645ULL;
        s     = s * MUL + inc;
        s_hi_ = (uint64_t)(s >> 64);
        s_lo_ = (uint64_t)s;
    }

    uint64_t xsl_rr() const {
        uint64_t xsl = s_hi_ ^ s_lo_;
        uint32_t rot = (uint32_t)(s_hi_ >> 58u);
        return (xsl >> rot) | (xsl << ((-rot) & 63u));
    }
};

// ---------------------------------------------------------------------------
// Lattice state codes — mirrors Python: EMPTY / FREE / DEPOSITED / SUBSTRATE / PASSIVATED
// ---------------------------------------------------------------------------
constexpr int8_t EMPTY      = 0;
constexpr int8_t FREE       = 1;
constexpr int8_t DEPOSITED  = 2;
constexpr int8_t SUBSTRATE  = 3;
constexpr int8_t PASSIVATED = 4;
constexpr int    NUM_STATES = 5;

struct KMCParams {
    int    Nx             = 40;
    int    Ny             = 25;
    double T               = 300.0;
    double d0               = 1.0e3;
    double e0               = -0.28;
    double e1               = -0.5;
    double nu_f             = 5.0e9;
    double nu_d             = 5.0e9;
    double nu_p             = 1.0e3;
    double kB               = 8.617333262145e-5;  // eV / K
    int    max_steps        = 4000000;
    double max_time         = 100.0;
    PCG64State pcg = {};  // default-constructed to seed=394583 values
    bool   periodic_x       = true;
    int    log_every        = 10000;
    int    snapshot_every   = 10000;
    bool   save_snapshots   = true;
    bool   save_npy         = true;
    std::string output_dir       = "kmc_output";
    std::string history_filename = "time_series.csv";
};

// -----------------------------------------------------------------------
// Native-only config file loader ("key = value" text file).
// -----------------------------------------------------------------------
#ifndef __EMSCRIPTEN__
static inline int toInt(const std::string& s, int def = 0) {
    try { return s.empty() ? def : std::stoi(s); } catch (...) { return def; }
}
static inline double toDouble(const std::string& s, double def = 0.0) {
    try { return s.empty() ? def : std::stod(s); } catch (...) { return def; }
}
static inline uint64_t toHex(const std::string& s, uint64_t def = 0) {
    try { return s.empty() ? def : std::stoull(s, nullptr, 16); } catch (...) { return def; }
}

KMCParams load_config(const std::string& path, KMCParams p = {}) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config file: " + path);
    std::string line;
    while (std::getline(f, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        auto trim = [](std::string& s) {
            size_t b = s.find_first_not_of(" \t\r\n");
            size_t e = s.find_last_not_of(" \t\r\n");
            s = (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
        };
        trim(key); trim(val);
        if      (key == "Nx")   p.Nx = toInt(val, p.Nx);
        else if (key == "Ny")   p.Ny = toInt(val, p.Ny);
        else if (key == "T")    p.T  = toDouble(val, p.T);
        else if (key == "d0")   p.d0 = toDouble(val, p.d0);
        else if (key == "e0")   p.e0 = toDouble(val, p.e0);
        else if (key == "e1")   p.e1 = toDouble(val, p.e1);
        else if (key == "nu_f") p.nu_f = toDouble(val, p.nu_f);
        else if (key == "nu_d") p.nu_d = toDouble(val, p.nu_d);
        else if (key == "nu_p") p.nu_p = toDouble(val, p.nu_p);
        else if (key == "max_steps") p.max_steps = toInt(val, p.max_steps);
        else if (key == "max_time")  p.max_time  = toDouble(val, p.max_time);
        else if (key == "pcg_state_hi") p.pcg.state_hi = toHex(val, p.pcg.state_hi);
        else if (key == "pcg_state_lo") p.pcg.state_lo = toHex(val, p.pcg.state_lo);
        else if (key == "pcg_inc_hi")   p.pcg.inc_hi   = toHex(val, p.pcg.inc_hi);
        else if (key == "pcg_inc_lo")   p.pcg.inc_lo   = toHex(val, p.pcg.inc_lo);
        else if (key == "periodic_x")     p.periodic_x     = (toInt(val, 1) != 0);
        else if (key == "log_every")      p.log_every      = toInt(val, p.log_every);
        else if (key == "snapshot_every") p.snapshot_every = toInt(val, p.snapshot_every);
        else if (key == "save_snapshots") p.save_snapshots = (toInt(val, 1) != 0);
        else if (key == "save_npy")       p.save_npy       = (toInt(val, 1) != 0);
        else if (key == "output_dir")     p.output_dir     = val;
        else if (key == "history_file")   p.history_filename = val;
    }
    return p;
}
#endif

// ---------------------------------------------------------------------------
// Fenwick Tree (Binary Indexed Tree) — mirrors Python FenwickTree class
// ---------------------------------------------------------------------------
class FenwickTree {
public:
    explicit FenwickTree(int size = 0) : size_(size), tree_(size + 1, 0.0) {}

    void reset(int size) {
        size_ = size;
        tree_.assign(size + 1, 0.0);
    }

    void update(int idx, double delta) {
        for (int i = idx + 1; i <= size_; i += i & -i)
            tree_[i] += delta;
    }

    double total() const {
        int i = size_;
        double s = 0.0;
        while (i > 0) { s += tree_[i]; i -= i & -i; }
        return s;
    }

    //smallest 0-based index whose prefix sum >= target.
    int find_prefix_index(double target) const {
        int idx = 0;
        int bit = 1;
        while (bit < size_) bit <<= 1;
        bit >>= 1;
        while (bit > 0) {
            int nxt = idx + bit;
            if (nxt <= size_ && tree_[nxt] < target) {
                target -= tree_[nxt];
                idx = nxt;
            }
            bit >>= 1;
        }
        return idx;
    }

private:
    int size_;
    std::vector<double> tree_;
};

// ---------------------------------------------------------------------------
// Event descriptor.
// kind: 0 = drop, 1 = hop, 2 = passivate
// For drop:      (dx,dy) = destination site (top row); (sx,sy) unused.
// For hop:       (sx,sy) = source site; (dx,dy) = raw destination (pre x-wrap).
// For passivate: (sx,sy) = (dx,dy) = the site itself.
// ---------------------------------------------------------------------------
struct Event {
    int8_t  kind;
    int16_t sx, sy;
    int16_t dx, dy;
};

struct HistoryRow {
    std::string label;
    int    step;
    double time;
    int    n_free;
    int    n_deposited;
    int    n_passivated;
    int    n_total;
    double total_rate;
};

// ---------------------------------------------------------------------------
// Main simulator class (mirrors ElectrodepositionKMC)
// ---------------------------------------------------------------------------
class ElectrodepositionKMC {
public:
    ~ElectrodepositionKMC() = default;

    explicit ElectrodepositionKMC(const KMCParams& p)
        : p_(p),
          rng_(p.pcg),
          lattice_(p.Ny * p.Nx, EMPTY),
          num_drop_(p.Nx),
          num_hop_(p.Nx * p.Ny * 6),
          num_passivate_(p.Nx * p.Ny),
          max_events_(p.Nx + p.Nx * p.Ny * 6 + p.Nx * p.Ny),
          event_rates_(max_events_, 0.0),
          ftree_(max_events_),
          idx_to_event_(max_events_)
    {
        if (p_.Nx < 1)  throw std::invalid_argument("Nx must be >= 1.");
        if (p_.Ny < 2)  throw std::invalid_argument("Ny must be >= 2.");
        if (p_.T <= 0)  throw std::invalid_argument("T must be positive.");

        // Substrate row (row 0); everything else starts EMPTY.
        for (int x = 0; x < p_.Nx; ++x) at(x, 0) = SUBSTRATE;

        // Interaction lookup table, indexed [from_type][to_type] — matches
        // the Python energy_lookup exactly, including the v3 conditional
        // DEPOSITED<->PASSIVATED bonds (handled in calc_local_energy, not
        // applied unconditionally here).
        memset(energy_lookup_, 0, sizeof(energy_lookup_));
        energy_lookup_[DEPOSITED][DEPOSITED]   = p_.e0;
        energy_lookup_[DEPOSITED][SUBSTRATE]   = p_.e1;
        energy_lookup_[SUBSTRATE][DEPOSITED]   = p_.e1;
        energy_lookup_[SUBSTRATE][SUBSTRATE]   = p_.e1;
        energy_lookup_[PASSIVATED][DEPOSITED]  = p_.e0;  // conditional, see calc_local_energy
        energy_lookup_[DEPOSITED][PASSIVATED]  = p_.e0;  // conditional, see calc_local_energy
        energy_lookup_[PASSIVATED][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][SUBSTRATE]  = p_.e1;
        energy_lookup_[SUBSTRATE][PASSIVATED]  = p_.e1;

#ifndef __EMSCRIPTEN__
        out_dir_ = fs::path(p_.output_dir);
        fs::create_directories(out_dir_);
        if (p_.save_snapshots) fs::create_directories(out_dir_ / "snapshots");
#endif

        setup_indices();
        rebuild_all_rates();

        record_history("initial");
#ifndef __EMSCRIPTEN__
        if (p_.save_snapshots) save_snapshot("initial");
        if (p_.save_npy)       save_lattice_npy("initial");
#endif
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    void run_cli() {
        while (step_ < p_.max_steps && time_ < p_.max_time) {
            if (!execute_step()) break;
            if (step_ % p_.log_every == 0) record_history("regular");
#ifndef __EMSCRIPTEN__
            if (p_.save_snapshots && step_ % p_.snapshot_every == 0) {
                char tag[32]; snprintf(tag, sizeof(tag), "step_%07d", step_);
                save_snapshot(tag);
            }
            if (p_.save_npy && step_ % p_.snapshot_every == 0) {
                char tag[32]; snprintf(tag, sizeof(tag), "step_%07d", step_);
                save_lattice_npy(tag);
            }
#endif
        }
        finalize_outputs();
    }

    bool execute_step() {
        double r_tot = ftree_.total();
        if (r_tot <= 0.0) return false;

        double u1 = std::max(rng_.next_double(), 1.0e-15);
        double dt = -std::log(u1) / r_tot;

        double u2     = std::max(rng_.next_double(), 1.0e-15);
        double target = u2 * r_tot;
        int    idx    = ftree_.find_prefix_index(target);

        const Event& ev = idx_to_event_[idx];
        std::vector<std::pair<int,int>> direct;

        if (ev.kind == 0) {                 // drop
            at(ev.dx, ev.dy) = FREE;
            direct.emplace_back(ev.dx, ev.dy);
        } else if (ev.kind == 2) {          // passivate
            at(ev.sx, ev.sy) = PASSIVATED;
            direct.emplace_back(ev.sx, ev.sy);
        } else {                             // hop
            int x1 = wrap_x(ev.dx);
            int y1 = ev.dy;
            int8_t atype = at(ev.sx, ev.sy);
            at(ev.sx, ev.sy) = EMPTY;
            at(x1, y1)       = atype;
            direct.emplace_back(ev.sx, ev.sy);
            direct.emplace_back(x1, y1);
        }

        auto relaxed = update_bonding_relaxation(direct);

        std::vector<std::pair<int,int>> all_changed = direct;
        all_changed.insert(all_changed.end(), relaxed.begin(), relaxed.end());
        refresh_local_rates(all_changed);

        time_ += dt;
        ++step_;
        return true;
    }

    int    step() const { return step_; }
    double time() const { return time_; }
    int    nx()   const { return p_.Nx; }
    int    ny()   const { return p_.Ny; }
    const int8_t* lattice_data() const { return lattice_.data(); }
    size_t lattice_size() const { return lattice_.size(); }

    struct Counts { int free, deposited, passivated, total; };
    Counts counts() const {
        int nf = 0, nd = 0, np = 0;
        for (auto v : lattice_) {
            if      (v == FREE)       ++nf;
            else if (v == DEPOSITED)  ++nd;
            else if (v == PASSIVATED) ++np;
        }
        return {nf, nd, np, nf + nd + np};
    }

private:
    // -----------------------------------------------------------------------
    // Lattice access
    // -----------------------------------------------------------------------
    int8_t& at(int x, int y)       { return lattice_[y * p_.Nx + x]; }
    int8_t  at(int x, int y) const { return lattice_[y * p_.Nx + x]; }

    int wrap_x(int x) const {
        if (p_.periodic_x) return ((x % p_.Nx) + p_.Nx) % p_.Nx;
        if (x >= 0 && x < p_.Nx) return x;
        return -1;
    }

    // Hexagonal neighbor deltas — row-parity dependent (pointy-top hex grid,
    // matches HEX_DX_EVEN/ODD, HEX_DY_EVEN/ODD in the Python file exactly).
    static void hex_deltas(int y, const int*& DX, const int*& DY) {
        static constexpr int DXE[6] = {1, -1, 0, 0, -1, -1};
        static constexpr int DYE[6] = {0,  0, 1, -1, 1, -1};
        static constexpr int DXO[6] = {1, -1, 0, 0,  1,  1};
        static constexpr int DYO[6] = {0,  0, 1, -1, 1, -1};
        if (y % 2 == 0) { DX = DXE; DY = DYE; }
        else            { DX = DXO; DY = DYO; }
    }

    // Calls f(nx, ny) for each of the up to 6 valid hex neighbors of (x,y).
    template<typename F>
    void for_each_neighbor(int x, int y, F&& f) const {
        const int* DX; const int* DY;
        hex_deltas(y, DX, DY);
        for (int k = 0; k < 6; ++k) {
            int xx = x + DX[k];
            int yy = y + DY[k];
            if (p_.periodic_x) xx = wrap_x(xx);
            else if (xx < 0 || xx >= p_.Nx) continue;
            if (yy < 0 || yy >= p_.Ny) continue;
            f(xx, yy);
        }
    }

    // -----------------------------------------------------------------------
    // Event indexing — mirrors Python _setup_indices / _build_event_arrays
    // -----------------------------------------------------------------------
    void setup_indices() {
        int top_y = p_.Ny - 1;

        //drop events: indices [0, Nx)
        for (int x = 0; x < p_.Nx; ++x)
            idx_to_event_[x] = Event{0, 0, 0, (int16_t)x, (int16_t)top_y};

        //hop events: indices [Nx, Nx + Nx*Ny*6)
        int base = num_drop_;
        for (int y = 0; y < p_.Ny; ++y) {
            const int* DX; const int* DY;
            hex_deltas(y, DX, DY);
            for (int x = 0; x < p_.Nx; ++x) {
                int site_off = (y * p_.Nx + x) * 6;
                for (int k = 0; k < 6; ++k) {
                    int idx = base + site_off + k;
                    idx_to_event_[idx] = Event{
                        1, (int16_t)x, (int16_t)y,
                        (int16_t)(x + DX[k]), (int16_t)(y + DY[k])
                    };
                }
            }
        }

        //passivation events: indices [Nx + Nx*Ny*6, Nx + Nx*Ny*6 + Nx*Ny)
        base = num_drop_ + num_hop_;
        for (int y = 0; y < p_.Ny; ++y)
            for (int x = 0; x < p_.Nx; ++x) {
                int idx = base + y * p_.Nx + x;
                idx_to_event_[idx] = Event{2, (int16_t)x, (int16_t)y, (int16_t)x, (int16_t)y};
            }
    }

    int drop_index(int x) const { return x; }
    int hop_base_index(int x, int y) const { return num_drop_ + (y * p_.Nx + x) * 6; }
    int passivate_index(int x, int y) const { return num_drop_ + num_hop_ + y * p_.Nx + x; }

    // -----------------------------------------------------------------------
    // Energetics — mirrors _nb_count_deposited_neighbors / _nb_calc_local_energy
    // -----------------------------------------------------------------------
    int count_deposited_neighbors(int x, int y) const {
        int n = 0;
        for_each_neighbor(x, y, [&](int nx, int ny) { if (at(nx, ny) == DEPOSITED) ++n; });
        return n;
    }

    bool has_empty_neighbor(int x, int y) const {
        bool found = false;
        for_each_neighbor(x, y, [&](int nx, int ny) { if (at(nx, ny) == EMPTY) found = true; });
        return found;
    }

    //desired mobile state (FREE or DEPOSITED) ignoring the site's own value —
    //mirrors _nb_desired_mobile_state.
    int8_t desired_mobile_state(int x, int y) const {
        int8_t result = FREE;
        for_each_neighbor(x, y, [&](int nx, int ny) {
            int8_t nbr = at(nx, ny);
            if (nbr == DEPOSITED || nbr == SUBSTRATE) result = DEPOSITED;
        });
        return result;
    }

    // DEPOSITED atom in that pair has >= 2 DEPOSITED neighbors of its own.
    // Mirrors _nb_calc_local_energy exactly.
    double calc_local_energy(int x, int y, int8_t atom_type) const {
        double e = 0.0;
        int dep_neighbors_at_site = -1;
        if (atom_type == DEPOSITED) dep_neighbors_at_site = count_deposited_neighbors(x, y);

        for_each_neighbor(x, y, [&](int nx, int ny) {
            int8_t nbr = at(nx, ny);
            if (atom_type == DEPOSITED && nbr == PASSIVATED) {
                if (dep_neighbors_at_site >= 2) e += energy_lookup_[DEPOSITED][PASSIVATED];
            } else if (atom_type == PASSIVATED && nbr == DEPOSITED) {
                if (count_deposited_neighbors(nx, ny) >= 2) e += energy_lookup_[PASSIVATED][DEPOSITED];
            } else {
                e += energy_lookup_[atom_type][nbr];
            }
        });
        return e;
    }

    // -----------------------------------------------------------------------
    // Event rates — mirrors _nb_get_event_rate
    // -----------------------------------------------------------------------
    double get_event_rate(const Event& ev) {
        if (ev.kind == 0) {                          // drop
            return (at(ev.dx, ev.dy) == EMPTY) ? p_.d0 : 0.0;
        }

        if (ev.kind == 2) {                          // passivate
            if (at(ev.sx, ev.sy) != DEPOSITED) return 0.0;
            return has_empty_neighbor(ev.sx, ev.sy) ? p_.nu_p : 0.0;
        }

        //hop
        int8_t atype = at(ev.sx, ev.sy);
        if (atype != FREE && atype != DEPOSITED) return 0.0;

        int x1 = wrap_x(ev.dx);
        int y1 = ev.dy;
        if (x1 == -1 || y1 < 0 || y1 >= p_.Ny) return 0.0;
        if (at(x1, y1) != EMPTY) return 0.0;

        double nu = (atype == FREE) ? p_.nu_f : p_.nu_d;
        double e_init = calc_local_energy(ev.sx, ev.sy, atype);

        //temporarily vacate source to evaluate destination energy correctly.
        at(ev.sx, ev.sy) = EMPTY;
        int8_t final_type = desired_mobile_state(x1, y1);
        double e_final = calc_local_energy(x1, y1, final_type);
        at(ev.sx, ev.sy) = atype;  // restore

        double dE = e_final - e_init;
        return nu * std::exp(-dE / (2.0 * p_.kB * p_.T));
    }

    void update_rate_at(int idx) {
        double new_rate = get_event_rate(idx_to_event_[idx]);
        double delta     = new_rate - event_rates_[idx];
        if (std::abs(delta) > 1.0e-18) {
            event_rates_[idx] = new_rate;
            ftree_.update(idx, delta);
        }
    }

    void rebuild_all_rates() {
        std::fill(event_rates_.begin(), event_rates_.end(), 0.0);
        ftree_.reset(max_events_);
        for (int i = 0; i < max_events_; ++i) update_rate_at(i);
    }

    // Refresh every event touching a 5x5 (Chebyshev radius-2) block around
    // each changed site — mirrors _nb_refresh_local_rates exactly (note: the
    // Python version does NOT restrict this to a Manhattan-distance diamond;
    // it's a full 5x5 square per changed site).
    void refresh_local_rates(const std::vector<std::pair<int,int>>& changed) {
        std::vector<char> visited(p_.Nx * p_.Ny, 0);
        std::vector<std::pair<int,int>> targets;
        targets.reserve(changed.size() * 25);

        int top_y = p_.Ny - 1;

        for (auto [cx, cy] : changed) {
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    int xx = cx + dx;
                    int yy = cy + dy;
                    if (p_.periodic_x) xx = wrap_x(xx);
                    else if (xx < 0 || xx >= p_.Nx) continue;
                    if (yy < 0 || yy >= p_.Ny) continue;
                    int lin = yy * p_.Nx + xx;
                    if (visited[lin]) continue;
                    visited[lin] = 1;
                    targets.emplace_back(xx, yy);
                }
            }
        }

        for (auto [x, y] : targets) {
            if (y == top_y) update_rate_at(drop_index(x));
            int base = hop_base_index(x, y);
            for (int k = 0; k < 6; ++k) update_rate_at(base + k);
            update_rate_at(passivate_index(x, y));
        }
    }

    std::vector<std::pair<int,int>> update_bonding_relaxation(
        const std::vector<std::pair<int,int>>& seeds)
    {
        std::vector<std::pair<int,int>> queue;
        std::vector<char> in_queue(p_.Nx * p_.Ny, 0);
        std::vector<std::pair<int,int>> changed;

        auto enqueue = [&](int x, int y) {
            int lin = y * p_.Nx + x;
            if (!in_queue[lin]) { in_queue[lin] = 1; queue.emplace_back(x, y); }
        };

        for (auto [sx, sy] : seeds) {
            enqueue(sx, sy);
            for_each_neighbor(sx, sy, [&](int nx, int ny) { enqueue(nx, ny); });
        }

        size_t head = 0;
        while (head < queue.size()) {
            auto [x, y] = queue[head++];
            int lin = y * p_.Nx + x;
            in_queue[lin] = 0;

            int8_t cur = at(x, y);
            if (cur != FREE && cur != DEPOSITED) continue;

            int8_t desired = desired_mobile_state(x, y);
            if (desired == cur) continue;

            at(x, y) = desired;
            changed.emplace_back(x, y);

            for_each_neighbor(x, y, [&](int nx, int ny) { enqueue(nx, ny); });
            enqueue(x, y);
        }
        return changed;
    }

    // -----------------------------------------------------------------------
    // Output helpers (native build only — WASM has no filesystem to write to)
    // -----------------------------------------------------------------------
    void record_history(const std::string& label) {
        auto c = counts();
        double tr = 0.0;
        for (double r : event_rates_) tr += r;
        history_.push_back({label, step_, time_, c.free, c.deposited, c.passivated, c.total, tr});
    }

#ifndef __EMSCRIPTEN__
    void write_history_csv() const {
        fs::path out = out_dir_ / p_.history_filename;
        std::ofstream f(out);
        if (!f) throw std::runtime_error("Cannot write history CSV: " + out.string());
        f << "label,step,time,free,deposited,passivated,total,total_rate\n";
        for (const auto& row : history_) {
            f << row.label << ',' << row.step << ','
              << std::scientific << std::setprecision(6) << row.time << ','
              << row.n_free << ',' << row.n_deposited << ',' << row.n_passivated << ','
              << row.n_total << ',' << row.total_rate << '\n';
        }
    }

    void save_snapshot(const std::string& tag) const {
        const int CELL_PX = std::max(8, std::min(24, 400 / std::max(p_.Nx, p_.Ny)));
        const int IMG_W = p_.Nx * CELL_PX;
        const int IMG_H = p_.Ny * CELL_PX;

        fs::path out = out_dir_ / "snapshots" / (tag + ".ppm");
        std::ofstream f(out, std::ios::binary);
        if (!f) { std::cerr << "Warning: cannot write snapshot " << out << '\n'; return; }

        f << "P6\n# LKMC | " << tag << " | step=" << step_
          << " | time=" << std::scientific << std::setprecision(3) << time_
          << " | T=" << p_.T << "K | Nx=" << p_.Nx << " Ny=" << p_.Ny << "\n"
          << "# Colors: white=empty  blue=free  orange=deposited  black=substrate  green=passivated\n"
          << IMG_W << ' ' << IMG_H << "\n255\n";

        struct RGB { uint8_t r, g, b; };
        static const RGB PAL[NUM_STATES] = {
            {0xff, 0xff, 0xff},   // EMPTY (white)
            {0x55, 0x99, 0xdd},   // FREE (steel blue)
            {0xdd, 0x88, 0x33},   // DEPOSITED (amber)
            {0x11, 0x11, 0x11},   // SUBSTRATE (black)
            {0x33, 0xaa, 0x55},   // PASSIVATED (green)
        };

        for (int ly = p_.Ny - 1; ly >= 0; --ly) {
            std::vector<uint8_t> row_buf(IMG_W * 3);
            for (int lx = 0; lx < p_.Nx; ++lx) {
                const RGB& c = PAL[(uint8_t)at(lx, ly)];
                for (int px = 0; px < CELL_PX; ++px) {
                    int base = (lx * CELL_PX + px) * 3;
                    row_buf[base + 0] = c.r;
                    row_buf[base + 1] = c.g;
                    row_buf[base + 2] = c.b;
                }
            }
            for (int py = 0; py < CELL_PX; ++py)
                f.write(reinterpret_cast<const char*>(row_buf.data()), row_buf.size());
        }
    }

    void save_lattice_npy(const std::string& tag) const {
        fs::path out = out_dir_ / ("lattice_" + tag + ".bin");
        std::ofstream f(out, std::ios::binary);
        if (!f) { std::cerr << "Warning: cannot write lattice bin " << out << '\n'; return; }
        int32_t header[2] = {(int32_t)p_.Ny, (int32_t)p_.Nx};
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
        f.write(reinterpret_cast<const char*>(lattice_.data()), lattice_.size());
    }
#endif

    void finalize_outputs() {
        record_history("final");
#ifndef __EMSCRIPTEN__
        std::string tag = "final_step_" + std::to_string(step_);
        if (p_.save_snapshots) save_snapshot(tag);
        if (p_.save_npy)       save_lattice_npy(tag);
        write_history_csv();
#endif
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------
    KMCParams p_;
    PCG64 rng_;

    std::vector<int8_t> lattice_;   // [y*Nx + x]
    double energy_lookup_[NUM_STATES][NUM_STATES];

    int num_drop_;
    int num_hop_;
    int num_passivate_;
    int max_events_;

    std::vector<double> event_rates_;
    FenwickTree          ftree_;
    std::vector<Event>   idx_to_event_;

    double time_ = 0.0;
    int    step_ = 0;

#ifndef __EMSCRIPTEN__
    fs::path out_dir_;
#endif
    std::vector<HistoryRow> history_;
};

// ---------------------------------------------------------------------------
// WASM-facing C API (very rough)
// ---------------------------------------------------------------------------
#ifdef __EMSCRIPTEN__

static ElectrodepositionKMC* wasm_sim = nullptr;

// File-scope state used to pass a pending PCG64 override into the next
// init_simulation() call.
static PCG64State pending_pcg_{};
static bool pending_pcg_override_ = false;

extern "C" {

// Create/replace the simulation with the given physical parameters. Uses the
// default PCG64 state (equivalent to numpy seed=394583) unless set_pcg_state
// is called beforehand to override it for this call.
EMSCRIPTEN_KEEPALIVE
void init_simulation(int Nx, int Ny, double T, double d0, double e0, double e1,
                      double nu_f, double nu_d, double nu_p,
                      int max_steps, double max_time, int periodic_x) {
    KMCParams params;
    params.Nx = Nx; params.Ny = Ny; params.T = T; params.d0 = d0;
    params.e0 = e0; params.e1 = e1; params.nu_f = nu_f; params.nu_d = nu_d;
    params.nu_p = nu_p; params.max_steps = max_steps; params.max_time = max_time;
    params.periodic_x = (periodic_x != 0);
    if (pending_pcg_override_) { params.pcg = pending_pcg_; pending_pcg_override_ = false; }

    delete wasm_sim;
    wasm_sim = new ElectrodepositionKMC(params);
}

// Override the PCG64 state used by the *next* init_simulation() call, so any
// numpy seed (not just the hardcoded default 394583) can be reproduced.
// Each 64-bit value is passed as two 32-bit halves (hi32, lo32).
// Use get_pcg64_state.py to compute these eight values for a given seed.
EMSCRIPTEN_KEEPALIVE
void set_pcg_state(uint32_t state_hi_hi, uint32_t state_hi_lo,
                    uint32_t state_lo_hi, uint32_t state_lo_lo,
                    uint32_t inc_hi_hi,   uint32_t inc_hi_lo,
                    uint32_t inc_lo_hi,   uint32_t inc_lo_lo) {
    pending_pcg_.state_hi = ((uint64_t)state_hi_hi << 32) | state_hi_lo;
    pending_pcg_.state_lo = ((uint64_t)state_lo_hi << 32) | state_lo_lo;
    pending_pcg_.inc_hi   = ((uint64_t)inc_hi_hi   << 32) | inc_hi_lo;
    pending_pcg_.inc_lo   = ((uint64_t)inc_lo_hi   << 32) | inc_lo_lo;
    pending_pcg_override_ = true;
}

// Advance the simulation by up to `steps` KMC events. Stops early if the
// event-rate table hits zero (nothing left that can happen).
EMSCRIPTEN_KEEPALIVE
void run_steps(int steps) {
    if (!wasm_sim) return;
    for (int i = 0; i < steps; ++i)
        if (!wasm_sim->execute_step()) break;
}

EMSCRIPTEN_KEEPALIVE
int get_step() { return wasm_sim ? wasm_sim->step() : 0; }

EMSCRIPTEN_KEEPALIVE
double get_time() { return wasm_sim ? wasm_sim->time() : 0.0; }

EMSCRIPTEN_KEEPALIVE
int get_nx() { return wasm_sim ? wasm_sim->nx() : 0; }

EMSCRIPTEN_KEEPALIVE
int get_ny() { return wasm_sim ? wasm_sim->ny() : 0; }

// Pointer into WASM linear memory where the lattice bytes live (one int8 per
// site, row-major, [y*Nx+x], values 0..4 — see EMPTY/FREE/DEPOSITED/
// SUBSTRATE/PASSIVATED). Read it from JS with:
//   Module.HEAPU8.subarray(ptr, ptr + Module._get_lattice_size())
// Re-fetch the pointer after any call that could reallocate memory
// (ALLOW_MEMORY_GROWTH can move the heap).
EMSCRIPTEN_KEEPALIVE
const int8_t* get_lattice_ptr() {
    static const int8_t empty = 0;
    return wasm_sim ? wasm_sim->lattice_data() : &empty;
}

EMSCRIPTEN_KEEPALIVE
int get_lattice_size() { return wasm_sim ? (int)wasm_sim->lattice_size() : 0; }

EMSCRIPTEN_KEEPALIVE
int get_free_count() { return wasm_sim ? wasm_sim->counts().free : 0; }

EMSCRIPTEN_KEEPALIVE
int get_deposited_count() { return wasm_sim ? wasm_sim->counts().deposited : 0; }

EMSCRIPTEN_KEEPALIVE
int get_passivated_count() { return wasm_sim ? wasm_sim->counts().passivated : 0; }

EMSCRIPTEN_KEEPALIVE
void cleanup_simulation() {
    delete wasm_sim;
    wasm_sim = nullptr;
}

} 

#endif  // __EMSCRIPTEN__

// ---------------------------------------------------------------------------
// Native CLI entry point (unused when compiled to WASM)
// ---------------------------------------------------------------------------
#ifndef __EMSCRIPTEN__
int main(int argc, char* argv[]) {
    KMCParams params;

    if (argc >= 2) {
        try {
            params = load_config(argv[1], params);
            std::cout << "Loaded config from: " << argv[1] << '\n';
        } catch (const std::exception& e) {
            std::cerr << "Config error: " << e.what() << '\n';
            return 1;
        }
    }

    std::cout << "LKMC Hexagonal Electrodeposition (C++ port)\n"
              << "  Lattice : " << params.Nx << " x " << params.Ny << '\n'
              << "  T       : " << params.T << " K\n"
              << "  d0      : " << params.d0 << "  e0 : " << params.e0
              << "  e1 : " << params.e1 << "  nu_p : " << params.nu_p << '\n'
              << "  max_steps : " << params.max_steps
              << "  max_time : " << params.max_time << " s\n"
              << std::flush;

    auto t0 = std::chrono::steady_clock::now();
    try {
        ElectrodepositionKMC sim(params);
        sim.run_cli();
        auto t1 = std::chrono::steady_clock::now();
        double ws = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "\nDone.  step=" << sim.step()
                  << "  time=" << std::scientific << std::setprecision(4) << sim.time()
                  << "  wall=" << std::fixed << std::setprecision(2) << ws << " s\n";
    } catch (const std::exception& e) {
        std::cerr << "Simulation error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
#endif