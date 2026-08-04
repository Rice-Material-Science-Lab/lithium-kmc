/**
 * lkmc-wasm.cpp  —  Lattice Kinetic Monte Carlo Electrodeposition Simulator
 *
 * WASM version of C++ port of LKMC_v2_commented_b.py.
 *
 *  * Build:
 * emcc lkmc-wasm.cpp -o public/lkmc-wasm.js -O3 -fexceptions -sINITIAL_MEMORY=268435456 -sEXPORT_ES6 -sMODULARIZE -sEXPORTED_FUNCTIONS="['_set_params','_update_simulation_params','_save_state','_get_save_state_len','_load_state','_peek_state_dimensions','_init_simulation','_run_steps','_play','_pause','_stop','_step_once','_playback_tick','_set_batch_size','_set_stats_interval','_get_stats_interval','_get_playback_state','_mark_carbon','_unmark_carbon','_finalize_carbon_placement','_set_carbon_species_energy','_get_carbon_species_grid','_run_batch','_get_batch_json','_get_lattice_data','_get_lattice','_get_lattice_size','_get_width','_get_height','_get_step','_get_wall_time','_get_time','_get_fill','_get_stats_json','_get_stats_json_len','_get_passivated','_get_terminated','_get_cell_coordination','_get_snapshot_count','_get_snapshot_step','_get_snapshot_lattice','_cleanup_simulation','_force_update_frontend','_malloc','_free']" -sEXPORTED_RUNTIME_METHODS="['ccall','cwrap','HEAP8','HEAPU8','HEAP32','HEAPF64','wasmMemory']" 
 * Exported WASM stuff:
 *   _set_params(int Nx, int Ny, double d0, double T, double e0, double e1, double nu_f, double nu_d, double nu_p, double e_pass, double nu_dp, double e_dp, int seed)
 *   _update_simulation_params(double d0, double T, double nu_f, double nu_d, double nu_p, double e_pass, double e0, double e1, double nu_dp, double e_dp)
 *   _get_lattice_data()
 *   _get_width()
 *   _get_height()
 *   _get_step()
 *   _get_time()
 *   _get_fill()
 *   _cleanup_simulation()
 *
 * Then, you should be able to use this on the web
 *
 * Emscripten docs (if you're confused):
 * https://emscripten.org/docs/index.html
 *
 *
 *  Params (all fields are optional; unrecognised keys are ignored):
 *
 *   Nx          = 40
 *   Ny          = 25
 *   T           = 300.0
 *   d0          = 1e3
 *   e0          = -0.2
 *   e1          = -0.5
 *   nu_f        = 5e9
 *   nu_d        = 1e9
 *   nu_p        = 1e6
 *   max_steps   = 400000
 *   max_time    = 100.0
 *   rng_seed    = 394583
 *   periodic_x  = 1
 *   log_every   = 1000
 *   snapshot_every = 10000
 *   save_snapshots = 1
 *   save_npy    = 1
 *   output_dir  = kmc_output
 *   history_file = time_series.csv
 *
 * Output files are written to output_dir/.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#ifndef __EMSCRIPTEN__
#include <filesystem>
#endif
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
extern "C"
{
    EM_JS(void, updateFrontend, (int step), {
        if (typeof window.updateSimulation === "function")
        {
            window.updateSimulation(step);
        }
    });

    EM_JS(void, notifySimTerminated, (), {
        if (typeof window.onSimulationTerminated === "function")
        {
            window.onSimulationTerminated();
        }
    });

    EM_JS(void, notifyBatchProgress, (int done, int total), {
        if (typeof window.onBatchRunProgress === "function")
        {
            window.onBatchRunProgress(done, total);
        }
    });
}
#endif
#ifndef __EMSCRIPTEN__
namespace fs = std::filesystem;
#endif

// ---------------------------------------------------------------------------
// PCG64 compatible output generator.
// Exact NumPy PCG64 reproduction requires importing the state/inc values.
//
// numpy uses PCG64 with a 128-bit LCG and XSL-RR output function.
// This class reproduces exactly the same sequence when initialized with
// the state/inc extracted from numpy via get_pcg64_state.py.
//
// Algorithm: advance state via 128-bit LCG, then apply XSL-RR output.
//   state = state * MUL + inc  (mod 2^128)
//   output = xsl_rr(new_state)  -> top 53 bits -> double in [0,1)
//
// To get the correct state/inc for a given Python seed, run:
//   python3 get_pcg64_state.py <seed>
// and paste the printed values into params.cfg as pcg_state_hi, etc.
// ---------------------------------------------------------------------------
struct PCG64State
{
    uint64_t state_hi = 0x50c3ed493ae78588ULL; // default: numpy seed=394583
    uint64_t state_lo = 0x2c8bef01c72f99e5ULL;
    uint64_t inc_hi = 0x71a5befeec2f5ccaULL;
    uint64_t inc_lo = 0x4df2b37d5d7aa1cbULL;

    // expand passed int into 256 bits needed
    void seed(uint64_t s)
    {
        auto splitmix64 = [](uint64_t &state) -> uint64_t
        {
            uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        };

        state_hi = splitmix64(s);
        state_lo = splitmix64(s);
        inc_hi = splitmix64(s);
        inc_lo = splitmix64(s) | 1ULL;
    }
};

class PCG64
{
public:
    explicit PCG64(const PCG64State &s)
        : s_hi_(s.state_hi), s_lo_(s.state_lo),
          i_hi_(s.inc_hi), i_lo_(s.inc_lo) {}

    double next_double()
    {
        advance();
        return (double)(xsl_rr() >> 11u) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t s_hi_, s_lo_, i_hi_, i_lo_;

    void advance()
    {
        // 128-bit LCG multiplier (same as numpy):
        // MUL = 0x2360ed051fc65da4_4385df649fccf645
        __uint128_t s = ((__uint128_t)s_hi_ << 64) | s_lo_;
        __uint128_t inc = ((__uint128_t)i_hi_ << 64) | i_lo_;
        const uint64_t MUL_HI = 0x2360ed051fc65da4ULL;
        const uint64_t MUL_LO = 0x4385df649fccf645ULL;
        const __uint128_t MUL =
            ((__uint128_t)MUL_HI << 64) | (__uint128_t)MUL_LO;
        s = s * MUL + inc;
        s_hi_ = (uint64_t)(s >> 64);
        s_lo_ = (uint64_t)s;
    }

    uint64_t xsl_rr() const
    {
        uint64_t xsl = s_hi_ ^ s_lo_;
        uint32_t rot = (uint32_t)(s_hi_ >> 58u);
        return (xsl >> rot) | (xsl << ((-rot) & 63u));
    }
};

// ---------------------------------------------------------------------------
// Lattice state codes
// ---------------------------------------------------------------------------
constexpr int8_t EMPTY = 0;
constexpr int8_t FREE = 1;
constexpr int8_t DEPOSITED = 2;
constexpr int8_t SUBSTRATE = 3;
constexpr int8_t PASSIVATED = 4;
constexpr int8_t CARBON = 5; // graphite anode site -- rigid, permanent, like SUBSTRATE
constexpr int MAX_CARBON_SPECIES = 4;
// ---------------------------------------------------------------------------
// Hexagonal lattice neighbour offsets (odd-r horizontal layout)
// ---------------------------------------------------------------------------

static constexpr int EVEN_DX[6] = {
    1, -1,
    0, -1,
    0, -1};

static constexpr int EVEN_DY[6] = {
    0, 0,
    -1, -1,
    1, 1};

static constexpr int ODD_DX[6] = {
    1, -1,
    1, 0,
    1, 0};

static constexpr int ODD_DY[6] = {
    0, 0,
    -1, -1,
    1, 1};

struct KMCParams
{
    int Nx = 100;
    int Ny = 100;
    double T = 300.0;
    double d0 = 1000.0;
    double e0 = -0.28;
    double e1 = -0.50;
    double carbon_species_energy[MAX_CARBON_SPECIES] = {-0.6, -0.4, -0.8, -0.3}; // eV, per anode species
    double nu_f = 5e7;
    double nu_d = 1e7;
    double nu_p = 1e6;
    double e_pass = 0.3; // eV — passivation activation energy barrier
    // Literature-cited SEI-forming decomposition barriers cluster
    // around 0.3-0.5 eV, close to (not far above) typical surface hop
    // barriers here (~0.15-0.3 eV) -- 0.3 keeps passivation reachable
    // and occasionally competitive rather than mathematically
    // unreachable given nu_p's slider range.
    double nu_dp = 1e5;  // de-passivation (SEI breakdown) attempt frequency
    double e_dp = 0.5;   // eV — de-passivation activation energy barrier;
    // higher than e_pass by default so passivation
    // is the dominant direction unless tuned otherwise
    // i love comments :]
    double kB = 8.617333262145e-5; // eV / K
    int max_steps = 400000;
    double max_time = 100.0;
    double stop_fill_fraction = -1.0;
    int stop_fill_total_sites = 0;
    int rng_seed = 394583;
    // PCG64 state — use get_pcg64_state.py to generate for any numpy seed.
    // Defaults match numpy.random.default_rng(394583).
    PCG64State pcg = {}; // default-constructed to seed=394583 values
    bool periodic_x = true;
    int log_every = 1000;
    int snapshot_every = 100;
    bool save_snapshots = true;
    bool save_npy = true;
    std::string output_dir = "kmc_output";
    std::string history_filename = "time_series.csv";
};

// Parse a simple "key = value" config file.
static inline int toInt(const std::string &s, int def = 0)
{
    try
    {
        return s.empty() ? def : std::stoi(s);
    }
    catch (...)
    {
        return def;
    }
}

static inline double toDouble(const std::string &s, double def = 0.0)
{
    try
    {
        return s.empty() ? def : std::stod(s);
    }
    catch (...)
    {
        return def;
    }
}

static inline uint64_t toHex(const std::string &s, uint64_t def = 0)
{
    try
    {
        return s.empty() ? def : std::stoull(s, nullptr, 16);
    }
    catch (...)
    {
        return def;
    }
}
KMCParams load_config(const std::string &path, KMCParams p = {})
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open config file: " + path);
    std::string line;
    while (std::getline(f, line))
    {
        // Strip comments and leading/trailing whitespace.
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line.erase(hash);
        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // Trim whitespace.
        auto trim = [](std::string &s)
        {
            size_t b = s.find_first_not_of(" \t\r\n");
            size_t e = s.find_last_not_of(" \t\r\n");
            s = (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
        };
        trim(key);
        trim(val);
        if (key == "Nx")
            p.Nx = toInt(val, p.Nx);
        else if (key == "Ny")
            p.Ny = toInt(val, p.Ny);
        else if (key == "T")
            p.T = toDouble(val, p.T);
        else if (key == "d0")
            p.d0 = toDouble(val, p.d0);
        else if (key == "e0")
            p.e0 = toDouble(val, p.e0);
        else if (key == "e1")
            p.e1 = toDouble(val, p.e1);
        else if (key == "nu_f")
            p.nu_f = toDouble(val, p.nu_f);
        else if (key == "nu_d")
            p.nu_d = toDouble(val, p.nu_d);
        else if (key == "nu_p")
            p.nu_p = toDouble(val, p.nu_p);
        else if (key == "max_steps")
            p.max_steps = toInt(val, p.max_steps);
        else if (key == "max_time")
            p.max_time = toDouble(val, p.max_time);

        else if (key == "pcg_state_hi")
            p.pcg.state_hi = toHex(val, p.pcg.state_hi);
        else if (key == "pcg_state_lo")
            p.pcg.state_lo = toHex(val, p.pcg.state_lo);
        else if (key == "pcg_inc_hi")
            p.pcg.inc_hi = toHex(val, p.pcg.inc_hi);
        else if (key == "pcg_inc_lo")
            p.pcg.inc_lo = toHex(val, p.pcg.inc_lo);

        else if (key == "periodic_x")
            p.periodic_x = (toInt(val, 1) != 0);
        else if (key == "log_every")
            p.log_every = toInt(val, p.log_every);
        else if (key == "snapshot_every")
            p.snapshot_every = toInt(val, p.snapshot_every);
        else if (key == "save_snapshots")
            p.save_snapshots = (toInt(val, 1) != 0);
        else if (key == "save_npy")
            p.save_npy = (toInt(val, 1) != 0);

        else if (key == "output_dir")
            p.output_dir = val;
        else if (key == "history_file")
            p.history_filename = val;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Fenwick Tree (Binary Indexed Tree) — mirrors Python FenwickTree class
// ---------------------------------------------------------------------------
class FenwickTree
{
public:
    explicit FenwickTree(int size)
        : size_(size), tree_(size + 1, 0.0) {}

    void reset(int size)
    {
        size_ = size;
        tree_.assign(size + 1, 0.0);
    }

    // Add delta to the element at 0-based index idx.
    void update(int idx, double delta)
    {
        for (int i = idx + 1; i <= size_; i += i & -i)
            tree_[i] += delta;
    }

    // Return total sum of all elements.
    double total() const
    {
        int i = size_;
        double s = 0.0;
        while (i > 0)
        {
            s += tree_[i];
            i -= i & -i;
        }
        return s;
    }

    // Return smallest 0-based index whose prefix sum >= target.
    int find_prefix_index(double target) const
    {
        int idx = 0;
        int bit = 1;
        while (bit < size_)
            bit <<= 1;
        bit >>= 1;
        while (bit > 0)
        {
            int nxt = idx + bit;
            if (nxt <= size_ && tree_[nxt] < target)
            {
                target -= tree_[nxt];
                idx = nxt;
            }
            bit >>= 1;
        }
        // Clamp: idx here is 0-based "last index whose prefix < target",
        // so the returned event index is idx, but it must never reach size_.
        if (idx >= size_)
            idx = size_ - 1;
        return idx;
    }

private:
    int size_;
    std::vector<double> tree_;
};

// ---------------------------------------------------------------------------
// Event descriptor (compact, avoids heap allocation per event)
// ---------------------------------------------------------------------------
struct Event
{
    bool is_drop;
    bool is_passivation;
    bool is_depassivation;
    int16_t sx, sy;
    int16_t dx, dy;
};

// for stats stuff

struct StatsRow
{
    int step;
    double time;
    int empty;
    int free;
    int deposited;
    int passivated;
    int substrate;
    double fill;
    double total_rate;
    double e_pass_used;   // debug: the e_pass value active at this step
    double nu_p_used;     // debug: the nu_p value active at this step
    double e_dp_used;     // debug: the e_dp value active at this step
    double nu_dp_used;    // debug: the nu_dp value active at this step
};

// ---------------------------------------------------------------------------
// History record
// ---------------------------------------------------------------------------
struct HistoryRow
{
    std::string label;
    int step;
    double time;
    int n_free;
    int n_deposited;
    int n_total;
    double total_rate;
};

// ---------------------------------------------------------------------------
// Main simulator class (mirrors ElectrodepositionKMC)
// ---------------------------------------------------------------------------
class ElectrodepositionKMC
{
public:
    ~ElectrodepositionKMC() = default;
    // Events per lattice site: 6 hop directions + 1 passivation + 1
    // de-passivation (SEI breakdown, reverts PASSIVATED -> DEPOSITED).
    static constexpr int kEventsPerSite = 8;

    explicit ElectrodepositionKMC(const KMCParams &p)
        : p_(p),
          rng_(p.pcg),
          lattice_(p.Ny * p.Nx, EMPTY),
          carbon_species_((size_t)p.Nx * p.Ny, -1),
          num_drop_(p.Nx),
          num_hop_(p.Nx * p.Ny * kEventsPerSite),
          max_events_(p.Nx + p.Nx * p.Ny * kEventsPerSite),
          event_rates_(p.Nx + p.Nx * p.Ny * kEventsPerSite, 0.0),
          ftree_(p.Nx + p.Nx * p.Ny * kEventsPerSite),
          idx_to_event_(p.Nx + p.Nx * p.Ny * kEventsPerSite)
    {
        // Validate.
        if (p_.Nx < 1)
            throw std::invalid_argument("Nx must be >= 1.");
        if (p_.Ny < 2)
            throw std::invalid_argument("Ny must be >= 2.");
        if (p_.T <= 0)
            throw std::invalid_argument("T must be positive.");

        wall_start_ = std::chrono::steady_clock::now();

        // Substrate row (row 0).
        for (int x = 0; x < p_.Nx; ++x)
            at(x, 0) = SUBSTRATE;

        // Build interaction lookup (indexed by [from_type][to_type]).
        // Matches the Python energy_lookup table exactly.
        memset(energy_lookup_, 0, sizeof(energy_lookup_));
        energy_lookup_[FREE][DEPOSITED] = p_.e0;
        energy_lookup_[DEPOSITED][FREE] = p_.e0;
        energy_lookup_[DEPOSITED][DEPOSITED] = p_.e0;
        energy_lookup_[FREE][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][FREE] = p_.e1;
        energy_lookup_[DEPOSITED][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][DEPOSITED] = p_.e1;
        energy_lookup_[SUBSTRATE][SUBSTRATE] = p_.e1;
        // PASSIVATED interacts exactly like DEPOSITED
        energy_lookup_[FREE][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][FREE] = p_.e0;
        energy_lookup_[DEPOSITED][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][DEPOSITED] = p_.e0;
        energy_lookup_[PASSIVATED][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][PASSIVATED] = p_.e1;
        // Li-C bond energy: carbon interacts with FREE/DEPOSITED/PASSIVATED
        // atoms the same way SUBSTRATE does via e1, but with its own
        // independently-tunable e_c value.

// Prepare output directory.
#ifndef __EMSCRIPTEN__
        out_dir_ = fs::path(p_.output_dir);
        fs::create_directories(out_dir_);
        if (p_.save_snapshots)
            fs::create_directories(out_dir_ / "snapshots");
#endif
        // Pre-size reusable scratch buffers once, up front, so steady-state
        // simulation never triggers a vector/map reallocation.
        refresh_visited_.assign((size_t)p_.Nx * p_.Ny, false);
        relax_in_queue_.assign((size_t)p_.Nx * p_.Ny, false);
        refresh_targets_.reserve(64);
        refresh_bfs_queue_.reserve(64);
        relax_queue_.reserve(64);
        relax_changed_.reserve(64);
        step_directly_changed_.reserve(4);
        step_all_changed_.reserve(64);

        // Build event index table.
        setup_indices();

        // Build initial rate table.
        rebuild_all_rates();

        // Record initial state.
        record_history("initial");
        record_stats();
#ifndef __EMSCRIPTEN__
        if (p_.save_snapshots)
            save_snapshot("initial");
        if (p_.save_npy)
            save_lattice_npy("initial");
#endif
    }

    const int8_t *lattice_data() const
    {
        return lattice_.data();
    }

    int width() const
    {
        return p_.Nx;
    }

    int height() const
    {
        return p_.Ny;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    void run_cli()
    {
        while (step_ < p_.max_steps && time_ < p_.max_time)
        {

            if (p_.stop_fill_fraction > 0.0)
            {
                if (fill_percentage() >= p_.stop_fill_fraction * 100.0)
                    break;
            }
            if (!execute_step())
                break;
            if (step_ % p_.log_every == 0)
            {
                record_history("regular");

                std::cout
                    << "step=" << step_
                    << " time=" << std::scientific << time_
                    << " fill=" << fill_percentage()
                    << "% rate=" << ftree_.total()
                    << "\n";
            }
#ifndef __EMSCRIPTEN__
            if (p_.save_snapshots && step_ % p_.snapshot_every == 0)
            {
                char tag[32];
                snprintf(tag, sizeof(tag), "step_%07d", step_);

                std::cout 
                    << "Saving snapshot: "
                    << tag
                    << "  step="
                    << step_
                    << "  time="
                    << std::scientific
                    << time_
                    << std::endl;

                save_snapshot(tag);
            }
            if (p_.save_npy && step_ % p_.snapshot_every == 0)
            {
                char tag[32];
                snprintf(tag, sizeof(tag), "step_%07d", step_);
                save_lattice_npy(tag);
            }
#endif
        }
        finalize_outputs();
    }

    int step() const { return step_; }
    bool is_terminated() const { return terminated_; }
    double time() const { return time_; }

    // Public wrapper so the frontend can inspect a clicked cell without
    // exposing the whole private energetics API.
    int get_coordination_at(int x, int y) const
    {
        if (x < 0 || x >= p_.Nx || y < 0 || y >= p_.Ny)
            return -1;
        return coordination_number(x, y);
    }

    int snapshot_count() const
    {
        return (int)lattice_snapshots_.size();
    }

    int snapshot_step_at(int idx) const
    {
        if (idx < 0 || idx >= (int)lattice_snapshots_.size())
            return -1;
        return lattice_snapshots_[idx].step;
    }

    const int8_t *snapshot_lattice_at(int idx) const
    {
        if (idx < 0 || idx >= (int)lattice_snapshots_.size())
            return nullptr;
        return lattice_snapshots_[idx].data.data();
    }
    // Serializes everything needed to resume a run: dimensions, step/time,
    // the raw lattice, and current params. Format is a flat buffer the
    // frontend can base64/store and hand back later.
    std::string serialize_state() const
    {
        std::ostringstream out(std::ios::binary);
        auto write = [&](const void *ptr, size_t n) {
            out.write(reinterpret_cast<const char *>(ptr), n);
        };
        int32_t nx = p_.Nx, ny = p_.Ny;
        write(&nx, sizeof(nx));
        write(&ny, sizeof(ny));
        write(&step_, sizeof(step_));
        write(&time_, sizeof(time_));
        write(lattice_.data(), lattice_.size());
        return out.str();
    }

    static bool deserialize_dimensions(const std::string &blob, int &nx, int &ny)
    {
        if (blob.size() < sizeof(int32_t) * 2)
            return false;
        memcpy(&nx, blob.data(), sizeof(int32_t));
        memcpy(&ny, blob.data() + sizeof(int32_t), sizeof(int32_t));
        return true;
    }

    // Restores lattice/step/time from a blob produced by serialize_state(),
    // matching this instance's Nx/Ny (caller must construct with the right
    // dimensions first via set_params + init_simulation).
    bool restore_state(const std::string &blob)
    {
        size_t header = sizeof(int32_t) * 2 + sizeof(step_) + sizeof(time_);
        if (blob.size() < header + lattice_.size())
            return false;

        size_t offset = sizeof(int32_t) * 2;
        memcpy(&step_, blob.data() + offset, sizeof(step_));
        offset += sizeof(step_);
        memcpy(&time_, blob.data() + offset, sizeof(time_));
        offset += sizeof(time_);
        memcpy(lattice_.data(), blob.data() + offset, lattice_.size());

        rebuild_all_rates();
        return true;
    }
    double wall_time() const
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start_
        ).count();
    }

private:
    // -----------------------------------------------------------------------
    // Lattice access helpers
    // -----------------------------------------------------------------------
    int8_t &at(int x, int y) { return lattice_[y * p_.Nx + x]; }
    int8_t at(int x, int y) const { return lattice_[y * p_.Nx + x]; }

    // Wrap x for periodic BC; returns -1 when out of bounds (non-periodic).
    int wrap_x(int x) const
    {
        if (p_.periodic_x)
            return ((x % p_.Nx) + p_.Nx) % p_.Nx;
        if (x >= 0 && x < p_.Nx)
            return x;
        return -1;
    }

    // Neighbour iteration helper: calls f(nx, ny) for each valid neighbour of (x,y).
    template <typename F>
    void for_each_neighbour(int x, int y, F &&f) const
    {

        const int *DX;
        const int *DY;

        if (y & 1)
        {
            DX = ODD_DX;
            DY = ODD_DY;
        }
        else
        {
            DX = EVEN_DX;
            DY = EVEN_DY;
        }

        for (int i = 0; i < 6; i++)
        {

            int nx = wrap_x(x + DX[i]);
            int ny = y + DY[i];

            if (nx == -1)
                continue;

            if (ny < 0 || ny >= p_.Ny)
                continue;

            f(nx, ny);
        }
    }

    // -----------------------------------------------------------------------
    // Event indexing (mirrors Python _setup_indices, drop_index, hop_base_index)
    // -----------------------------------------------------------------------
    void setup_indices()
    {
        int top_y = p_.Ny - 1;
        // Drop events: indices [0, Nx)
        for (int x = 0; x < p_.Nx; ++x)
        {
            idx_to_event_[x] =
            {
                true,
                false,
                false,
                0,
                0,
                (int16_t)x,
                (int16_t)top_y
            };
        }
        int base = num_drop_;
        for (int y = 0; y < p_.Ny; ++y)
        {
            for (int x = 0; x < p_.Nx; ++x)
            {
                int site_off = (y * p_.Nx + x) * kEventsPerSite;
                for (int d = 0; d < 6; ++d)
                {
                    int idx = base + site_off + d;

                    const int *DX = (y & 1) ? ODD_DX : EVEN_DX;
                    const int *DY = (y & 1) ? ODD_DY : EVEN_DY;

                    idx_to_event_[idx] =
                    {
                        false,
                        false,
                        false,
                        (int16_t)x,
                        (int16_t)y,
                        (int16_t)(x + DX[d]),
                        (int16_t)(y + DY[d])
                    };
                }

                // passivation event
                idx_to_event_[base + site_off + 6] =
                {
                    false,
                    true,
                    false,
                    (int16_t)x,
                    (int16_t)y,
                    0,
                    0
                };

                // de-passivation (SEI breakdown) event
                idx_to_event_[base + site_off + 7] =
                {
                    false,
                    false,
                    true,
                    (int16_t)x,
                    (int16_t)y,
                    0,
                    0
                };
            }
        }
    }

    int drop_index(int x) const { return x; }
    int hop_base_index(int x, int y) const
    {
        return num_drop_ + (y * p_.Nx + x) * kEventsPerSite;
    }

    // -----------------------------------------------------------------------
    // Energetics
    // -----------------------------------------------------------------------
    double carbon_bond_energy_at(int nx, int ny) const
    {
        int8_t sp = carbon_species_[ny * p_.Nx + nx];
        if (sp < 0 || sp >= MAX_CARBON_SPECIES)
            sp = 0;
        return p_.carbon_species_energy[sp];
    }

    double calc_local_energy(int x, int y, int8_t atom_type) const
    {
        double e = 0.0;

        int coord = 0;

        for_each_neighbour(x, y, [&](int nx, int ny)
        {
            int8_t n = at(nx, ny);

            if (n == DEPOSITED || n == PASSIVATED)
                coord++;

            if (n == CARBON)
                e += carbon_bond_energy_at(nx, ny);
            else
                e += energy_lookup_[atom_type][n];
        });

        return e;
    }

    int coordination_number(int x, int y) const
    {
        int coord = 0;

        for_each_neighbour(x, y, [&](int nx, int ny)
        {
            int8_t s = at(nx, ny);

            if (s == DEPOSITED || s == PASSIVATED)
                coord++;
        });

        return coord;
    }

    double bond_energy(int coordination) const
    {
        // Simple version

        return p_.e0 * coordination;
    }

    double get_event_rate(const Event &ev) const
    {
        if (ev.is_drop)
        {
            int x1 = ev.dx;
            int y1 = ev.dy;

            if (at(x1, y1) != EMPTY)
                return 0.0;

            int coord = coordination_number(x1, y1);

            double E_dep = 0.15 - 0.02 * coord;

            if(E_dep < 0.02)
                E_dep = 0.02;

            return p_.d0 *
                exp(
                    -E_dep /
                    (p_.kB * p_.T)
                );
        }

        int x0 = ev.sx, y0 = ev.sy;
        int8_t atype = at(x0, y0);
        // passivation event
        if(ev.is_passivation)
        {
            if (atype != DEPOSITED)
                return 0.0;
            // Passivation only occurs on exposed deposited atoms
            bool exposed = false;
            int empty_neighbors = 0;
            int passivated_neighbors = 0;
            for_each_neighbour(x0, y0, [&](int nx, int ny)
                               {
                int8_t n = at(nx, ny);
                if(n == EMPTY) {
                    exposed = true;
                    empty_neighbors++;
                }
                if(n == PASSIVATED) {
                    passivated_neighbors++;
                } });
            if (!exposed)
                return 0.0;
            // Apply the same Boltzmann suppression as hop/drop events so
            // passivation competes fairly with growth instead of
            // dominating regardless of temperature.
            // empty_neighbors ranges 0-6 on this hex lattice; divide by 6
            // so the exposure factor is a genuine 0-1 fraction instead of
            // occasionally exceeding 1.0 (the old /3.0 let a fully exposed
            double exposure_fraction = empty_neighbors / 6.0;

            // Self-limiting SEI growth: real SEI formation is
            // diffusion-limited once a passivation layer already exists
            // nearby (Peled's SEI model -- roughly parabolic/sqrt(t)
            // growth, not unbounded). Model this as an extra activation
            // barrier per already-passivated neighbor: a bare exposed
            // atom still passivates readily (an initial monolayer forms
            // fast, matching how real SEI forms within seconds of
            // electrolyte contact), but further growth into an
            // already-coated region is exponentially suppressed instead
            // of running away to cover the whole lattice.
            static constexpr double kSeiGrowthBarrier = 0.15; // eV/neighbor
            double local_barrier =
                p_.e_pass + kSeiGrowthBarrier * passivated_neighbors;

            return p_.nu_p *
                exp(-local_barrier / (p_.kB * p_.T)) *
                exposure_fraction;
        }
        // de-passivation (SEI breakdown) event, reverts an exposed
        // PASSIVATED atom back to DEPOSITED. Mirrors the passivation
        // rate shape (same exposure-fraction weighting) but with its
        // own independently-tunable frequency/barrier so growth and
        // breakdown can be tuned separately.
        if (ev.is_depassivation)
        {
            if (atype != PASSIVATED)
                return 0.0;
            bool exposed = false;
            int empty_neighbors = 0;
            for_each_neighbour(x0, y0, [&](int nx, int ny)
                               {
                if(at(nx,ny) == EMPTY) {
                    exposed = true;
                    empty_neighbors++;
                } });
            if (!exposed)
                return 0.0;
            double exposure_fraction = empty_neighbors / 6.0;
            return p_.nu_dp *
                exp(-p_.e_dp / (p_.kB * p_.T)) *
                exposure_fraction;
        }
        if (atype != FREE && atype != DEPOSITED)
            return 0.0;

        int x1 = wrap_x(ev.dx);
        int y1 = ev.dy;
        if (x1 == -1 || y1 < 0 || y1 >= p_.Ny)
            return 0.0;
        if (at(x1, y1) != EMPTY)
            return 0.0;

        double nu = (atype == FREE) ? p_.nu_f : p_.nu_d;
        double e_init = calc_local_energy(x0, y0, atype);
        int coord_initial = coordination_number(x0, y0);

        // Temporarily remove atom to compute destination energy.
        int8_t old = const_cast<ElectrodepositionKMC *>(this)->at(x0,y0);

        const_cast<ElectrodepositionKMC *>(this)->at(x0,y0) = EMPTY;

        double e_final = calc_local_energy(x1,y1,atype);
        const_cast<ElectrodepositionKMC *>(this)->at(x0,y0) = old;
        if(!std::isfinite(e_final) || !std::isfinite(e_init))
        {
            const_cast<ElectrodepositionKMC *>(this)->at(x0,y0)=old;
            return 0.0;
        }
        int coord_final = coordination_number(x1, y1);
        const_cast<ElectrodepositionKMC *>(this)->at(x0, y0) = atype;

        double barrier =
            0.15 + 
            0.03 * std::max(0, coord_initial - coord_final);

        double energy_penalty = std::max(0.0, e_final - e_init);

        double rate =
            nu *
            exp(
                -(barrier + energy_penalty)
                /
                (p_.kB * p_.T)
            );

        return rate;
    }

    void update_rate_at(int idx)
    {
        double new_rate = get_event_rate(idx_to_event_[idx]);

        if(!std::isfinite(new_rate))
        {
#ifndef __EMSCRIPTEN__
            printf(
                "INVALID RATE idx=%d rate=%e\n",
                idx,
                new_rate
            );
#endif
            new_rate = 0.0;
        }

        if(new_rate > 1e100)
        {
#ifndef __EMSCRIPTEN__
            printf(
                "RATE TOO LARGE idx=%d rate=%e\n",
                idx,
                new_rate
            );
#endif
            new_rate = 1e100;
        }
        double delta = new_rate - event_rates_[idx];
        if (std::abs(delta) > 1.0e-18)
        {
            event_rates_[idx] = new_rate;
            ftree_.update(idx, delta);
        }
    }

    void rebuild_all_rates()
    {
        std::fill(event_rates_.begin(), event_rates_.end(), 0.0);
        ftree_.reset(max_events_);
        for (int i = 0; i < max_events_; ++i)
            update_rate_at(i);
    }

    // -----------------------------------------------------------------------
    // Local rate refresh (radius-2 neighbourhood — mirrors refresh_local_rates)
    // -----------------------------------------------------------------------
    void refresh_local_rates(const std::vector<std::pair<int, int>> &changed)
    {
        // Reuse member scratch buffers instead of allocating fresh
        // containers every step (prevents WASM heap fragmentation on
        // very long runs).
        refresh_targets_.clear();
        std::fill(refresh_visited_.begin(), refresh_visited_.end(), false);
        refresh_dist_.clear();
        refresh_bfs_queue_.clear();

        for (auto [sx, sy] : changed)
        {
            refresh_bfs_queue_.emplace_back(sx, sy);
            refresh_dist_[sy * p_.Nx + sx] = 0;
        }

        size_t head = 0;
        while (head < refresh_bfs_queue_.size())
        {
            auto [x, y] = refresh_bfs_queue_[head++];
            int d = refresh_dist_[y * p_.Nx + x];
            int linear = y * p_.Nx + x;
            if (!refresh_visited_[linear])
            {
                refresh_visited_[linear] = true;
                refresh_targets_.emplace_back(x, y);
            }
            if (d == 2)
                continue;
            for_each_neighbour(x, y, [&](int nx, int ny)
                               {
                    int key = ny * p_.Nx + nx;
                    if (!refresh_dist_.count(key)) {
                        refresh_dist_[key] = d + 1;
                        refresh_bfs_queue_.emplace_back(nx, ny);
                    } });
        }

        int top_y = p_.Ny - 1;
        for (auto [x, y] : refresh_targets_)
        {
            if (y == top_y)
                update_rate_at(drop_index(x));
            int base = hop_base_index(x, y);
            for (int d = 0; d < kEventsPerSite; ++d)
                update_rate_at(base + d);
        }
    }

    // Writes results into member relax_changed_ instead of returning by
    // value, so no vector is allocated/copied on every step.
    int8_t desired_bond_state(int x, int y) const
    {
        int8_t st = at(x, y);
        if (st != FREE && st != DEPOSITED)
            return st;
        bool bonded = false;
        for_each_neighbour(x, y, [&](int nx, int ny)
                           {
                if (at(nx, ny) == DEPOSITED ||
                    at(nx, ny) == PASSIVATED ||
                    at(nx, ny) == SUBSTRATE ||
                    at(nx, ny) == CARBON)
                    bonded = true; });
        return bonded ? DEPOSITED : FREE;
    }

    void update_bonding_relaxation(
        const std::vector<std::pair<int, int>> &seeds)
    {
        relax_queue_.clear();
        std::fill(relax_in_queue_.begin(), relax_in_queue_.end(), false);
        relax_changed_.clear();

        // Seed with each site and its direct neighbours.
        auto enqueue = [&](int x, int y)
        {
            int lin = y * p_.Nx + x;
            if (!relax_in_queue_[lin])
            {
                relax_in_queue_[lin] = true;
                relax_queue_.emplace_back(x, y);
            }
        };
        for (auto [sx, sy] : seeds)
        {
            enqueue(sx, sy);
            for_each_neighbour(sx, sy, [&](int nx, int ny)
                               { enqueue(nx, ny); });
        }

        size_t head = 0;
        while (head < relax_queue_.size())
        {
            auto [x, y] = relax_queue_[head++];
            int lin = y * p_.Nx + x;
            relax_in_queue_[lin] = false; // allow re-enqueue if needed

            int8_t cur = at(x, y);
            if (cur != FREE && cur != DEPOSITED)
                continue;

            int8_t desired = desired_bond_state(x, y);
            if (desired == cur)
                continue;

            at(x, y) = desired;
            relax_changed_.emplace_back(x, y);

            // Re-enqueue neighbours and self.
            for_each_neighbour(x, y, [&](int nx, int ny)
                               { enqueue(nx, ny); });
            enqueue(x, y);
        }
    }

public:
    // -----------------------------------------------------------------------
    // KMC step (mirrors execute_step)
    // -----------------------------------------------------------------------
    bool execute_step()
    {
        if (terminated_)
            return false;

        if(parameters_changed_)
        {
            rebuild_all_rates();
            parameters_changed_ = false;
        }
        double r_tot = ftree_.total();
#ifndef __EMSCRIPTEN__
        if(step_ % 100 == 0)
        {
            printf(
                "STEP %d RATE %.6e TIME %.6e FILL %.3f%%\n",
                step_,
                r_tot,
                time_,
                fill_percentage()
            );
        }
#endif

        if (r_tot <= 0.0)
        {
            int empty = 0;
            int free = 0;
            int deposited = 0;
            int passivated = 0;

            for (auto v : lattice_)
            {
                if (v == EMPTY) empty++;
                else if (v == FREE) free++;
                else if (v == DEPOSITED) deposited++;
                else if (v == PASSIVATED) passivated++;
            }

            // printf (stdout) instead of std::cerr (stderr) -- Emscripten's
            // default glue code routes stderr straight to console.error,
            // which is what was showing up as a JS console error.
            printf(
                "KMC STOP: total event rate reached zero step=%d time=%f EMPTY=%d FREE=%d DEPOSITED=%d PASSIVATED=%d\n",
                step_, time_, empty, free, deposited, passivated
            );

            terminated_ = true;
            return false;
        }

        // Time increment.
        double u1 = std::max(rng_.next_double(), 1.0e-15);
        double dt = -std::log(u1) / r_tot;

        // Select event.
        double u2 = std::max(rng_.next_double(), 1.0e-15);
        double target = u2 * r_tot;
        int idx = ftree_.find_prefix_index(target);
        if (idx < 0) idx = 0;
        if (idx >= max_events_) idx = max_events_ - 1;

        const Event &ev = idx_to_event_[idx];
        step_directly_changed_.clear();

        if (ev.is_drop)
        {
            int x1 = ev.dx, y1 = ev.dy;
            at(x1, y1) = FREE;
            step_directly_changed_.emplace_back(x1, y1);
        }
        else
        {
            int x0 = ev.sx, y0 = ev.sy;

            if (ev.is_passivation)
            {
                if (at(x0, y0) == DEPOSITED)
                {
                    at(x0, y0) = PASSIVATED;
                    step_directly_changed_.emplace_back(x0, y0);
                }
            }
            else if (ev.is_depassivation)
            {
                if (at(x0, y0) == PASSIVATED)
                {
                    at(x0, y0) = DEPOSITED;
                    step_directly_changed_.emplace_back(x0, y0);
                }
            }
            else
            {
                int x1 = wrap_x(ev.dx);
                int y1 = ev.dy;

                if (x1 == -1 || y1 < 0 || y1 >= p_.Ny)
                {
                    // Stale/invalid event (rate table out of sync) — skip safely.
                    return true;
                }

                int8_t atype = at(x0, y0);
                at(x0, y0) = EMPTY;
                at(x1, y1) = atype;
                step_directly_changed_.emplace_back(x0, y0);
                step_directly_changed_.emplace_back(x1, y1);
            }
        }

        update_bonding_relaxation(step_directly_changed_); // fills relax_changed_

        // Merge changed sets (reused buffer, no fresh allocation).
        step_all_changed_.clear();
        step_all_changed_.insert(step_all_changed_.end(),
                                  step_directly_changed_.begin(), step_directly_changed_.end());
        step_all_changed_.insert(step_all_changed_.end(),
                                  relax_changed_.begin(), relax_changed_.end());
        refresh_local_rates(step_all_changed_);
        time_ += dt;
        ++step_;
        if(step_ % stats_interval_ == 0)
        {
            record_stats();
        }
        return true;
    }
    void update_params(
        double d0,
        double T,
        double nu_f,
        double nu_d,
        double nu_p,
        double e_pass,
        double e0,
        double e1,
        double nu_dp,
        double e_dp
    )
    {
        p_.d0 = d0;
        p_.T = T;
        p_.nu_f = nu_f;
        p_.nu_d = nu_d;
        p_.nu_p = nu_p;
        // Floor at 0.05 eV: passivation should always carry some
        // suppression relative to growth. Prevents a caller (or a future
        // plumbing bug) from silently disabling the Boltzmann barrier by
        // passing 0, which would let passivation dominate unrealistically.
        p_.e_pass = std::max(e_pass, 0.05);
        p_.e0 = e0;
        p_.e1 = e1;
        p_.nu_dp = nu_dp;
        p_.e_dp = std::max(e_dp, 0.05); // same floor rationale as e_pass

        // energy_lookup_ entries that depend on p_.e0 / p_.e1 must also be
        // refreshed live. CARBON energies are looked up per-cell via
        // carbon_species_energy[] instead, and are set independently via
        // set_carbon_species_energy().
        energy_lookup_[FREE][DEPOSITED] = p_.e0;
        energy_lookup_[DEPOSITED][FREE] = p_.e0;
        energy_lookup_[DEPOSITED][DEPOSITED] = p_.e0;
        energy_lookup_[FREE][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][FREE] = p_.e1;
        energy_lookup_[DEPOSITED][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][DEPOSITED] = p_.e1;
        energy_lookup_[SUBSTRATE][SUBSTRATE] = p_.e1;
        energy_lookup_[FREE][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][FREE] = p_.e0;
        energy_lookup_[DEPOSITED][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][DEPOSITED] = p_.e0;
        energy_lookup_[PASSIVATED][PASSIVATED] = p_.e0;
        energy_lookup_[PASSIVATED][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][PASSIVATED] = p_.e1;

        // Important: old rates are now invalid
        parameters_changed_ = true;
    }
    int passivated_count() const
    {
        int count = 0;

        for (auto v : lattice_)
        {
            if (v == PASSIVATED)
                count++;
        }

        return count;
    }

    // Marks a single lattice cell as a graphite anode site. Call this for
    // each user-drawn carbon cell after init_simulation(), then call
    // finalize_carbon_placement() once at the end -- rebuilding the rate
    // table per-cell would be wasteful for a large drawn region.
    void set_carbon_site(int x, int y, int species)
    {
        if (x < 0 || x >= p_.Nx || y < 0 || y >= p_.Ny)
            return;
        at(x, y) = CARBON;
        int sp = std::max(0, std::min(species, MAX_CARBON_SPECIES - 1));
        carbon_species_[y * p_.Nx + x] = (int8_t)sp;
    }

    // Reverts a cell that was previously marked carbon back to EMPTY.
    // Used for live toggle-off; only safe to call on a cell that is
    // actually CARBON (a site with an atom already bonded to it is left
    // alone rather than silently deleting that atom).
    void unset_carbon_site(int x, int y)
    {
        if (x < 0 || x >= p_.Nx || y < 0 || y >= p_.Ny)
            return;
        if (at(x, y) == CARBON)
        {
            at(x, y) = EMPTY;
            carbon_species_[y * p_.Nx + x] = -1;
        }
    }

    void set_carbon_species_energy(int species, double energy)
    {
        if (species < 0 || species >= MAX_CARBON_SPECIES)
            return;
        p_.carbon_species_energy[species] = energy;
        parameters_changed_ = true;
    }

    const int8_t *carbon_species_data() const
    {
        return carbon_species_.data();
    }

    void finalize_carbon_placement()
    {
        rebuild_all_rates();
    }
    // Guarantees a JSON-safe number token -- never emits nan/inf, which
    // are invalid JSON and break JSON.parse on the frontend.
    static double json_safe(double v)
    {
        return std::isfinite(v) ? v : 0.0;
    }

    std::string get_stats_json() const
    {
        std::ostringstream json;

        json << "[";    

        for(size_t i = 0; i < stats_history_.size(); i++)
        {
            const auto &s = stats_history_[i];

            json << "{"
                << "\"step\":" << s.step << ","
                << "\"time\":" << s.time << ","
                << "\"empty\":" << s.empty << ","
                << "\"free\":" << s.free << ","
                << "\"deposited\":" << s.deposited << ","
                << "\"passivated\":" << s.passivated << ","
                << "\"substrate\":" << s.substrate << ","
                << "\"fill\":" << s.fill << ","
                << "\"total_rate\":" << json_safe(s.total_rate) << ","
                << "\"e_pass_used\":" << json_safe(s.e_pass_used) << ","
                << "\"nu_p_used\":" << json_safe(s.nu_p_used) << ","
                << "\"e_dp_used\":" << json_safe(s.e_dp_used) << ","
                << "\"nu_dp_used\":" << json_safe(s.nu_dp_used)
                << "}";

            if(i + 1 < stats_history_.size())
                json << ",";
        }

        json << "]";

        return json.str();
    }
    double fill_percentage() const
    {
        int deposited = 0;

        for (auto v : lattice_)
        {
            if (
                v == FREE ||
                v == DEPOSITED ||
                v == PASSIVATED
            )
                deposited++;
        }

        int total_sites = p_.Nx * p_.Ny;

        return 100.0 * deposited / total_sites;
    }
    void play()
    {
        playback_state_ = PlaybackState::PLAYING;
    }

    void pause()
    {
        playback_state_ = PlaybackState::PAUSED;
    }

    void stop()
    {
        playback_state_ = PlaybackState::STOPPED;
        stop_requested_ = true;
    }

    void step_once()
    {
        execute_step();
    }

    void playback_tick()
    {
        if(playback_state_ != PlaybackState::PLAYING)
            return;

        for(int i = 0; i < playback_batch_; i++)
        {
            if(stop_requested_)
            {
                stop_requested_ = false;
                playback_state_ = PlaybackState::STOPPED;
                break;
            }

            if(!execute_step())
            {
                playback_state_ = PlaybackState::STOPPED;
                break;
            }
        }
    }

    void set_batch_size(int batch)
    {
        playback_batch_ = std::max(1, batch);
    }

    void set_stats_interval(int interval)
    {
        stats_interval_ = std::max(1, interval);
    }

    int stats_interval() const { return stats_interval_; }

    int playback_state() const
    {
        return static_cast<int>(playback_state_);
    }

private:
    // -----------------------------------------------------------------------
    // Output helpers
    // -----------------------------------------------------------------------
    struct Counts
    {
        int free, dep, total;
    };

    void record_stats()
    {
        int empty = 0;
        int free = 0;
        int deposited = 0;
        int passivated = 0;
        int substrate = 0;

        for(auto v : lattice_)
        {
            switch(v)
            {
                case EMPTY:
                    empty++;
                    break;

                case FREE:
                    free++;
                    break;

                case DEPOSITED:
                    deposited++;
                    break;

                case PASSIVATED:
                    passivated++;
                    break;

                case SUBSTRATE:
                    substrate++;
                    break;
            }
        }

        double total_rate = 0.0;

        for(auto r : event_rates_)
            total_rate += r;

        stats_history_.push_back({
            step_,
            time_,
            empty,
            free,
            deposited,
            passivated,
            substrate,
            fill_percentage(),
            total_rate,
            p_.e_pass,
            p_.nu_p,
            p_.e_dp,
            p_.nu_dp
        });

        // Prevent unbounded memory growth on indefinite/very long runs:
        // once the cap is hit, halve the history by keeping every other
        // row and double the sampling interval going forward. This keeps
        // memory bounded while preserving the overall shape of the curve.
        if (stats_history_.size() > kMaxStatsRows)
        {
            std::vector<StatsRow> compacted;
            compacted.reserve(stats_history_.size() / 2 + 1);
            for (size_t i = 0; i < stats_history_.size(); i += 2)
                compacted.push_back(stats_history_[i]);
            stats_history_.swap(compacted);
            stats_interval_ *= 2;
        }

        // Snapshot the full lattice at the same cadence as stats, so the
        // frontend can scrub back through history rather than only seeing
        // the live state. Bounded the same way -- halve and keep going
        // once the cap is hit.
        lattice_snapshots_.push_back({step_, time_, lattice_});
        if (lattice_snapshots_.size() > kMaxLatticeSnapshots)
        {
            std::vector<LatticeSnapshot> compacted;
            compacted.reserve(lattice_snapshots_.size() / 2 + 1);
            for (size_t i = 0; i < lattice_snapshots_.size(); i += 2)
                compacted.push_back(std::move(lattice_snapshots_[i]));
            lattice_snapshots_.swap(compacted);
        }
    }

    Counts counts() const
    {
        int nf = 0, nd = 0;
        for (auto v : lattice_)
        {
            if (v == FREE)
                ++nf;
            else if (v == DEPOSITED || v == PASSIVATED)
                ++nd;
        }
        return {nf, nd, nf + nd};
    }

    void record_history(const std::string &label)
    {
        auto [nf, nd, nt] = counts();
        double tr = 0.0;
        for (double r : event_rates_)
            tr += r;
        history_.push_back({label, step_, time_, nf, nd, nt, tr});
    }

    void append_results_csv()
    {

        std::ofstream f("AllResults.csv", std::ios::app);

        if (!f)
        {
            throw std::runtime_error("Cannot open AllResults.csv");
        }

        // Write header if file is empty
        if (f.tellp() == 0)
        {
            f << "d0,T,e0,pv,vf,vd,seed,percentage,Steps,Time\n";
        }

        f
            << p_.d0 << ","
            << p_.T << ","
            << p_.e0 << ","
            << p_.nu_p << ","
            << p_.nu_f << ","
            << p_.nu_d << ","
            << p_.rng_seed << ","
            << fill_percentage() << ","
            << step_ << ","
            << time_
            << "\n";
    }

#ifndef __EMSCRIPTEN__
    void write_history_csv() const
    {
        fs::path out = out_dir_ / p_.history_filename;
        std::ofstream f(out);
        if (!f)
            throw std::runtime_error("Cannot write history CSV: " + out.string());
        f << "label,step,time,free,deposited,total_mobile_plus_deposited,total_rate\n";
        for (const auto &row : history_)
        {
            f << row.label << ','
              << row.step << ','
              << std::scientific << std::setprecision(6) << row.time << ','
              << row.n_free << ','
              << row.n_deposited << ','
              << row.n_total << ','
              << row.total_rate << '\n';
        }
    }
#endif

// Write a colour PPM (P6) snapshot scaled up so each lattice cell is CELL_PX pixels.
// PPM is supported by most image viewers, GIMP, Photoshop, and IrfanView without plugins.
// Colors: black=EMPTY  steel-blue=FREE  amber=DEPOSITED  dark-grey=SUBSTRATE
#ifndef __EMSCRIPTEN__
    void save_snapshot(const std::string &tag) const
    {
        const int CELL_PX = std::max(8, std::min(24, 400 / std::max(p_.Nx, p_.Ny)));
        const int IMG_W = p_.Nx * CELL_PX;
        const int IMG_H = p_.Ny * CELL_PX;

        fs::path out = out_dir_ / "snapshots" / (tag + ".ppm");
        std::ofstream f(out, std::ios::binary);
        if (!f)
        {
            std::cerr << "Warning: cannot write snapshot " << out << '\n';
            return;
        }

        // PPM header with metadata comment.
        f << "P6\n"
          << "# LKMC | " << tag
          << " | step=" << step_
          << " | time=" << std::scientific << std::setprecision(3) << time_
          << " | T=" << p_.T << "K | Nx=" << p_.Nx << " Ny=" << p_.Ny << "\n"
          << "# Colors: black=empty  blue=free  orange=deposited  darkgrey=substrate\n"
          << IMG_W << ' ' << IMG_H << "\n255\n";

        struct RGB
        {
            uint8_t r, g, b;
        };
        static const RGB PAL[6] = {
            {0x11, 0x11, 0x11}, // EMPTY
            {0x55, 0x99, 0xdd}, // FREE       (steel blue)
            {0xdd, 0x88, 0x33}, // DEPOSITED  (amber)
            {0x22, 0x22, 0x22}, // SUBSTRATE  (dark grey)
            {0x99, 0x33, 0xcc}, // PASSIVATED (purple)
            {0xdd, 0x22, 0x22}, // CARBON     (red)
        };

        // Write rows top-to-bottom (lattice row 0 = substrate = bottom of image).
        for (int ly = p_.Ny - 1; ly >= 0; --ly)
        {
            std::vector<uint8_t> row_buf(IMG_W * 3);
            for (int lx = 0; lx < p_.Nx; ++lx)
            {
                const RGB &c = PAL[(uint8_t)at(lx, ly)];
                for (int px = 0; px < CELL_PX; ++px)
                {
                    int base = (lx * CELL_PX + px) * 3;
                    row_buf[base + 0] = c.r;
                    row_buf[base + 1] = c.g;
                    row_buf[base + 2] = c.b;
                }
            }
            for (int py = 0; py < CELL_PX; ++py)
                f.write(reinterpret_cast<const char *>(row_buf.data()), row_buf.size());
        }
    }
#endif

// Write the raw lattice as a simple binary file: 4-byte header (Ny, Nx),
// then Ny*Nx int8 values in row-major order (row 0 = substrate).
#ifndef __EMSCRIPTEN__
    void save_lattice_npy(const std::string &tag) const
    {
        fs::path out = out_dir_ / ("lattice_" + tag + ".bin");
        std::ofstream f(out, std::ios::binary);
        if (!f)
        {
            std::cerr << "Warning: cannot write lattice bin " << out << '\n';
            return;
        }
        int32_t header[2] = {(int32_t)p_.Ny, (int32_t)p_.Nx};
        f.write(reinterpret_cast<const char *>(header), sizeof(header));
        f.write(reinterpret_cast<const char *>(lattice_.data()), lattice_.size());
    }
#endif

    void finalize_outputs()
    {
        record_history("final");

#ifndef __EMSCRIPTEN__
        std::ostringstream tag_stream;

        tag_stream
            << "d" << p_.d0
            << "T" << p_.T
            << "e" << p_.e0
            << "pv" << p_.nu_p
            << "vf" << p_.nu_f
            << "vd" << p_.nu_d
            << "s" << p_.rng_seed
            << "p" << fill_percentage();

        std::string tag = tag_stream.str();

        if (p_.save_snapshots)
        {
            std::cout 
                << "Saving final snapshot step="
                << step_
                << " time="
                << time_
                << std::endl;

            save_snapshot(tag);
        }
        if (p_.save_npy)
            save_lattice_npy(tag);
        write_history_csv();
        append_results_csv();
#endif
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------
    KMCParams p_;
    PCG64 rng_;
    std::chrono::steady_clock::time_point wall_start_;

    std::vector<int8_t> lattice_; // [y*Nx + x]
    std::vector<int8_t> carbon_species_; // [y*Nx + x], -1 = not carbon
    double energy_lookup_[6][6];
    int num_drop_;
    int num_hop_;
    int max_events_;

    std::vector<double> event_rates_;
    FenwickTree ftree_;
    std::vector<Event> idx_to_event_;

    // Reusable scratch buffers — avoids per-step heap allocation/free churn,
    // which fragments the WASM heap over very long (unbounded) runs and
    // eventually causes an allocation failure / crash.
    std::vector<bool> refresh_visited_;
    std::vector<std::pair<int, int>> refresh_targets_;
    std::unordered_map<int, int> refresh_dist_;
    std::vector<std::pair<int, int>> refresh_bfs_queue_;

    std::vector<bool> relax_in_queue_;
    std::vector<std::pair<int, int>> relax_queue_;
    std::vector<std::pair<int, int>> relax_changed_;

    std::vector<std::pair<int, int>> step_directly_changed_;
    std::vector<std::pair<int, int>> step_all_changed_;

    static constexpr size_t kMaxStatsRows = 5000;
    static constexpr size_t kMaxLatticeSnapshots = 300;

    struct LatticeSnapshot
    {
        int step;
        double time;
        std::vector<int8_t> data;
    };
    std::vector<LatticeSnapshot> lattice_snapshots_;

    double time_ = 0.0;
    int step_ = 0;
    bool parameters_changed_ = false;
    enum class PlaybackState
    {
        STOPPED,
        PLAYING,
        PAUSED
    };

    PlaybackState playback_state_ = PlaybackState::STOPPED;

    bool stop_requested_ = false;
    bool terminated_ = false;

    int playback_batch_ = 100;

    int stats_interval_ = 10000;
#ifndef __EMSCRIPTEN__
    fs::path out_dir_;
#endif
    std::vector<HistoryRow> history_;
    // Stores chart data every 10k steps
    std::vector<StatsRow> stats_history_;
};

#ifdef __EMSCRIPTEN__

extern "C"
{

    static ElectrodepositionKMC *wasm_sim = nullptr;
    static KMCParams wasm_params = KMCParams{};

    EMSCRIPTEN_KEEPALIVE
    void set_params(
        int Nx,
        int Ny,
        double d0,
        double T,
        double e0,
        double e1,
        double nu_f,
        double nu_d,
        double nu_p,
        double e_pass,
        double nu_dp,
        double e_dp,
        int seed)
    {
        wasm_params.Nx = Nx;
        wasm_params.Ny = Ny;
        wasm_params.d0 = d0;
        wasm_params.T = T;
        wasm_params.e0 = e0;
        wasm_params.e1 = e1;
        wasm_params.nu_f = nu_f;
        wasm_params.nu_d = nu_d;
        // enable passivation
        wasm_params.nu_p = nu_p;
        // Same floor as update_params(), applied here too since
        // set_params() is the other entry point that sets e_pass.
        wasm_params.e_pass = std::max(e_pass, 0.05);
        wasm_params.nu_dp = nu_dp;
        wasm_params.e_dp = std::max(e_dp, 0.05);
        wasm_params.rng_seed = seed;

        wasm_params.pcg.seed((uint64_t)seed);
    }

    EMSCRIPTEN_KEEPALIVE
    void set_carbon_species_energy(int species, double energy)
    {
        // Applies to whatever's currently active: the running sim if one
        // exists, and wasm_params so a fresh init_simulation() picks it up.
        if (species >= 0 && species < MAX_CARBON_SPECIES)
            wasm_params.carbon_species_energy[species] = energy;
        if (wasm_sim)
            wasm_sim->set_carbon_species_energy(species, energy);
    }

    EMSCRIPTEN_KEEPALIVE
    const int8_t *get_carbon_species_grid()
    {
        return wasm_sim ? wasm_sim->carbon_species_data() : nullptr;
    }

    EMSCRIPTEN_KEEPALIVE
   void update_simulation_params(
        double d0,
        double T,
        double nu_f,
        double nu_d,
        double nu_p,
        double e_pass,
        double e0,
        double e1,
        double nu_dp,
        double e_dp
    )
    {
        if (!wasm_sim)
            return;

        wasm_sim->update_params(
            d0,
            T,
            nu_f,
            nu_d,
            nu_p,
            e_pass,
            e0,
            e1,
            nu_dp,
            e_dp
        );
    }

    EMSCRIPTEN_KEEPALIVE
    void init_simulation()
    {

        if (wasm_sim != nullptr)
        {
            delete wasm_sim;
            wasm_sim = nullptr;
        }

        wasm_sim = new ElectrodepositionKMC(wasm_params);
    }

    EMSCRIPTEN_KEEPALIVE
    void mark_carbon(int x, int y, int species)
    {
        if (wasm_sim)
            wasm_sim->set_carbon_site(x, y, species);
    }

    EMSCRIPTEN_KEEPALIVE
    void unmark_carbon(int x, int y)
    {
        if (wasm_sim)
            wasm_sim->unset_carbon_site(x, y);
    }

    EMSCRIPTEN_KEEPALIVE
    void finalize_carbon_placement()
    {
        if (wasm_sim)
            wasm_sim->finalize_carbon_placement();
    }

    EMSCRIPTEN_KEEPALIVE
    void run_steps(int steps)
    {
        if (!wasm_sim)
        {
            printf("CRITICAL ERROR: wasm_sim is NULL at the start of run_steps!\n");
            return;
        }

        for (int i = 0; i < steps; i++)
        {
            bool success = false;
            try
            {
                success = wasm_sim->execute_step();
            }
            catch (...)
            {
                printf("CRITICAL ERROR: Exception thrown inside execute_step()!\n");
                break;
            }

            if (!success)
                break;
        }
#ifdef __EMSCRIPTEN__
        if (wasm_sim != nullptr)
        {
            // Update the frontend on every run_steps() call rather than
            // gating on step % 10000. The frontend already batches its own
            // calls (currently 1000 simulated steps per animation frame in
            // tick()), so this naturally ties visual refresh rate to that
            // loop instead of double-throttling on top of it. This matters
            // for transient states like FREE: an atom is only FREE for a
            // single step before update_bonding_relaxation() promotes it to
            // DEPOSITED, so infrequent snapshots almost always miss it.
            updateFrontend(wasm_sim->step());
            if (wasm_sim->is_terminated())
                notifySimTerminated();
        }
        else
        {
            printf("CRITICAL ERROR: wasm_sim became NULL right before updateFrontend!\n");
        }
#endif
    }

    EMSCRIPTEN_KEEPALIVE
    const int8_t *get_lattice_data()
    {
        if (!wasm_sim)
        {
            printf("CRITICAL ERROR: wasm_sim is NULL during get_lattice_data!\n");
            return nullptr;
        }
        return wasm_sim->lattice_data();
    }

    EMSCRIPTEN_KEEPALIVE
    int get_width()
    {
        return wasm_sim ? wasm_sim->width() : 0;
    }

    // Shared buffer so the frontend can query the EXACT byte length before
    // reading WASM memory, instead of guessing a fixed max length (which
    // can read past the end of the heap and throw an uncaught RangeError
    // in JS -- crashing the whole WASM instance mid-call).
    static std::string g_stats_json_buf;

    EMSCRIPTEN_KEEPALIVE
    const char *get_stats_json()
    {
        if (!wasm_sim)
        {
            g_stats_json_buf = "{}";
            return g_stats_json_buf.c_str();
        }

        g_stats_json_buf = wasm_sim->get_stats_json();

        return g_stats_json_buf.c_str();
    }

    EMSCRIPTEN_KEEPALIVE
    int get_stats_json_len()
    {
        return (int)g_stats_json_buf.size();
    }

    EMSCRIPTEN_KEEPALIVE
    int get_height()
    {
        return wasm_sim ? wasm_sim->height() : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    const int8_t *get_lattice()
    {
        return wasm_sim ? wasm_sim->lattice_data() : nullptr;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_lattice_size()
    {
        return wasm_sim ? wasm_sim->width() * wasm_sim->height() : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_fill()
    {
        return wasm_sim ? wasm_sim->fill_percentage() : 0.0;
    }
    EMSCRIPTEN_KEEPALIVE
    int get_passivated()
    {
        return wasm_sim ? wasm_sim->passivated_count() : 0;
    }
    EMSCRIPTEN_KEEPALIVE
    int get_step()
    {
        if (!wasm_sim)
            return 0;
        return wasm_sim->step();
    }

    EMSCRIPTEN_KEEPALIVE
    double get_time()
    {
        if (!wasm_sim)
            return 0.0;
        return wasm_sim->time();
    }

    EMSCRIPTEN_KEEPALIVE
    double get_wall_time()
    {
        if (!wasm_sim)
            return 0.0;
        return wasm_sim->wall_time();
    }

    EMSCRIPTEN_KEEPALIVE
    void play()
    {
        if(wasm_sim)
            wasm_sim->play();
    }

    EMSCRIPTEN_KEEPALIVE
    void pause()
    {
        if(wasm_sim)
            wasm_sim->pause();
    }

    EMSCRIPTEN_KEEPALIVE
    void stop()
    {
        if(wasm_sim)
            wasm_sim->stop();
    }

    EMSCRIPTEN_KEEPALIVE
    void step_once()
    {
        if(wasm_sim)
            wasm_sim->step_once();
    }

    EMSCRIPTEN_KEEPALIVE
    void playback_tick()
    {
        if(wasm_sim)
            wasm_sim->playback_tick();
    }

    EMSCRIPTEN_KEEPALIVE
    void set_batch_size(int batch)
    {
        if(wasm_sim)
            wasm_sim->set_batch_size(batch);
    }

    EMSCRIPTEN_KEEPALIVE
    void set_stats_interval(int interval)
    {
        if(wasm_sim)
            wasm_sim->set_stats_interval(interval);
    }

    EMSCRIPTEN_KEEPALIVE
    int get_stats_interval()
    {
        return wasm_sim ? wasm_sim->stats_interval() : -1;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_playback_state()
    {
        return wasm_sim ? wasm_sim->playback_state() : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_terminated()
    {
        return (wasm_sim && wasm_sim->is_terminated()) ? 1 : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_cell_coordination(int x, int y)
    {
        return wasm_sim ? wasm_sim->get_coordination_at(x, y) : -1;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_snapshot_count()
    {
        return wasm_sim ? wasm_sim->snapshot_count() : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_snapshot_step(int idx)
    {
        return wasm_sim ? wasm_sim->snapshot_step_at(idx) : -1;
    }

    EMSCRIPTEN_KEEPALIVE
    const int8_t *get_snapshot_lattice(int idx)
    {
        return wasm_sim ? wasm_sim->snapshot_lattice_at(idx) : nullptr;
    }
    
    static std::string g_save_state_buf;

    EMSCRIPTEN_KEEPALIVE
    const char *save_state()
    {
        if (!wasm_sim)
        {
            g_save_state_buf.clear();
            return g_save_state_buf.c_str();
        }
        g_save_state_buf = wasm_sim->serialize_state();
        return g_save_state_buf.c_str();
    }

    EMSCRIPTEN_KEEPALIVE
    int get_save_state_len()
    {
        return (int)g_save_state_buf.size();
    }

    // Caller must have already called set_params (with matching Nx/Ny read
    // from the blob) and init_simulation before calling this.
    EMSCRIPTEN_KEEPALIVE
    int load_state(const char *data, int len)
    {
        if (!wasm_sim || !data || len <= 0)
            return 0;
        std::string blob(data, len);
        return wasm_sim->restore_state(blob) ? 1 : 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int peek_state_dimensions(const char *data, int len, int *out_nx, int *out_ny)
    {
        if (!data || len <= 0)
            return 0;
        std::string blob(data, len);
        int nx = 0, ny = 0;
        if (!ElectrodepositionKMC::deserialize_dimensions(blob, nx, ny))
            return 0;
        if (out_nx) *out_nx = nx;
        if (out_ny) *out_ny = ny;
        return 1;
    }

    static std::string g_batch_json_buf;

    EMSCRIPTEN_KEEPALIVE
    void run_batch(
        double *d0_arr,
        double *T_arr,
        double *e0_arr,
        double *e1_arr,
        int num_runs,
        int nx,
        int ny,
        int steps_per_run,
        int base_seed)
    {
        std::ostringstream json;
        json << "[";

        for (int i = 0; i < num_runs; i++)
        {
            KMCParams p = wasm_params; // inherit current defaults (carbon energies, etc.)
            p.Nx = nx;
            p.Ny = ny;
            p.d0 = d0_arr[i];
            p.T = T_arr[i];
            p.e0 = e0_arr[i];
            p.e1 = e1_arr[i];
            p.rng_seed = base_seed + i;
            p.pcg.seed((uint64_t)p.rng_seed);

            auto t0 = std::chrono::steady_clock::now();
            ElectrodepositionKMC sim(p);
            int actual_steps = 0;
            for (int s = 0; s < steps_per_run; s++)
            {
                if (!sim.execute_step())
                    break;
                actual_steps++;
            }
            double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0
            ).count();

            json << "{"
                 << "\"index\":" << i << ","
                 << "\"d0\":" << p.d0 << ","
                 << "\"T\":" << p.T << ","
                 << "\"e0\":" << p.e0 << ","
                 << "\"e1\":" << p.e1 << ","
                 << "\"steps_run\":" << actual_steps << ","
                 << "\"final_step\":" << sim.step() << ","
                 << "\"final_time\":" << sim.time() << ","
                 << "\"fill_pct\":" << sim.fill_percentage() << ","
                 << "\"passivated\":" << sim.passivated_count() << ","
                 << "\"terminated\":" << (sim.is_terminated() ? 1 : 0) << ","
                 << "\"wall_time\":" << wall
                 << "}";
            if (i + 1 < num_runs)
                json << ",";

#ifdef __EMSCRIPTEN__
            notifyBatchProgress(i + 1, num_runs);
#endif
        }

        json << "]";
        g_batch_json_buf = json.str();
    }

    EMSCRIPTEN_KEEPALIVE
    const char *get_batch_json()
    {
        return g_batch_json_buf.c_str();
    }

    EMSCRIPTEN_KEEPALIVE
    void cleanup_simulation()
    {
        delete wasm_sim;
        wasm_sim = nullptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void force_update_frontend()
    {
        if (wasm_sim)
        {
            updateFrontend(wasm_sim->step());
        }
    }
}

#endif

#ifndef __EMSCRIPTEN__
// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    KMCParams params;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--d0")
        {
            params.d0 = std::stod(argv[++i]);
        }
        else if (arg == "--T")
        {
            params.T = std::stod(argv[++i]);
        }
        else if (arg == "--e0")
        {
            params.e0 = std::stod(argv[++i]);
        }
        else if (arg == "--pv")
        {
            params.nu_p = std::stod(argv[++i]);
        }
        else if (arg == "--vf")
        {
            params.nu_f = std::stod(argv[++i]);
        }
        else if (arg == "--vd")
        {
            params.nu_d = std::stod(argv[++i]);
        }
        else if (arg == "--maxStep")
        {
            params.max_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--maxTime")
        {
            params.max_time = std::stod(argv[++i]);
        }
        else if (arg == "--snapshotEvery")
        {
            params.snapshot_every = std::stoi(argv[++i]);
        }
        else if (arg == "--seed")
        {
            params.rng_seed = std::stoi(argv[++i]);
        }
        else if (arg == "--p")
        {
            double percentage = std::stod(argv[++i]);

            if (percentage < 1 || percentage > 100)
                throw std::invalid_argument("--p must be between 1 and 100");

            params.stop_fill_fraction = percentage / 100.0;
            params.stop_fill_total_sites = params.Nx * params.Ny;
        }
        else if (arg == "--config")
        {
            params = load_config(argv[++i], params);
        }
    }

    std::cout << "LKMC Electrodeposition  (C++ port)\n"
              << "  Lattice : " << params.Nx << " x " << params.Ny << '\n'
              << "  T       : " << params.T << " K\n"
              << "  d0      : " << params.d0 << "  e0 : " << params.e0
              << "  e1 : " << params.e1 << '\n'
              << "  max_steps : " << params.max_steps
              << "  max_time : " << params.max_time << " s\n"
              << "  pcg state: " << std::hex << params.pcg.state_hi << "_" << params.pcg.state_lo << std::dec << '\n'
              << std::flush;

    auto t0 = std::chrono::steady_clock::now();

    try
    {
        ElectrodepositionKMC sim(params);
        sim.run_cli();

        auto t1 = std::chrono::steady_clock::now();
        double ws = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\nDone.  step=" << sim.step()
                  << "  time=" << std::scientific << std::setprecision(4) << sim.time()
                  << "  wall=" << std::fixed << std::setprecision(2) << ws << " s\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Simulation error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
#endif