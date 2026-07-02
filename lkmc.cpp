/**
 * lkmc.cpp  —  Lattice Kinetic Monte Carlo Electrodeposition Simulator
 *
 * C++ port of LKMC_v2_commented_b.py.
 *
 * Build (single command, no Makefile needed):
 *   g++ -O2 -std=c++17 -o lkmc lkmc.cpp
 *
 * Run:
 *   ./lkmc                  # uses built-in default parameters
 *  
 *
 * Config file format (all fields are optional; unrecognised keys are ignored):
 *   Nx          = 40
 *   Ny          = 25
 *   T           = 300.0
 *   d0          = 1e3
 *   e0          = -0.2
 *   e1          = -0.5
 *   nu_f        = 5e9
 *   nu_d        = 1e9
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// PCG64 — identical to numpy.random.default_rng() draw sequence.
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

private:
    uint64_t s_hi_, s_lo_, i_hi_, i_lo_;

    void advance() {
        // 128-bit LCG multiplier (same as numpy):
        // MUL = 0x2360ed051fc65da4_4385df649fccf645
        __uint128_t s   = ((__uint128_t)s_hi_ << 64) | s_lo_;
        __uint128_t inc = ((__uint128_t)i_hi_ << 64) | i_lo_;
        const __uint128_t MUL =
            ((__uint128_t)0x2360ed051fc65da4ULL << 64) | 0x4385df649fccf645ULL;
        s    = s * MUL + inc;
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
// Lattice state codes
// ---------------------------------------------------------------------------
constexpr int8_t EMPTY     = 0;
constexpr int8_t FREE      = 1;
constexpr int8_t DEPOSITED = 2;
constexpr int8_t SUBSTRATE = 3;


struct KMCParams {
    int    Nx             = 40;
    int    Ny             = 25;
    double T              = 300.0;
    double d0             = 1.0e3;
    double e0             = -0.2;
    double e1             = -0.5;
    double nu_f           = 5.0e9;
    double nu_d           = 1.0e9;
    double kB             = 8.617333262145e-5;  // eV / K
    int    max_steps      = 400000;
    double max_time       = 100.0;
    // PCG64 state — use get_pcg64_state.py to generate for any numpy seed.
    // Defaults match numpy.random.default_rng(394583).
    PCG64State pcg = {};  // default-constructed to seed=394583 values
    bool   periodic_x     = true;
    int    log_every      = 1000;
    int    snapshot_every = 10000;
    bool   save_snapshots = true;
    bool   save_npy       = true;
    std::string output_dir      = "kmc_output";
    std::string history_filename = "time_series.csv";
};

// Parse a simple "key = value" config file.
static inline int toInt(const std::string& s, int def = 0) {
    try { return s.empty() ? def : std::stoi(s); }
    catch (...) { return def; }
}

static inline double toDouble(const std::string& s, double def = 0.0) {
    try { return s.empty() ? def : std::stod(s); }
    catch (...) { return def; }
}

static inline uint64_t toHex(const std::string& s, uint64_t def = 0) {
    try { return s.empty() ? def : std::stoull(s, nullptr, 16); }
    catch (...) { return def; }
}
KMCParams load_config(const std::string& path, KMCParams p = {}) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config file: " + path);
    std::string line;
    while (std::getline(f, line)) {
        // Strip comments and leading/trailing whitespace.
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // Trim whitespace.
        auto trim = [](std::string& s) {
            size_t b = s.find_first_not_of(" \t\r\n");
            size_t e = s.find_last_not_of(" \t\r\n");
            s = (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
        };
        trim(key); trim(val);
        if (key == "Nx")              p.Nx = toInt(val, p.Nx);
        else if (key == "Ny")         p.Ny = toInt(val, p.Ny);
        else if (key == "T")          p.T  = toDouble(val, p.T);
        else if (key == "d0")         p.d0 = toDouble(val, p.d0);
        else if (key == "e0")         p.e0 = toDouble(val, p.e0);
        else if (key == "e1")         p.e1 = toDouble(val, p.e1);
        else if (key == "nu_f")       p.nu_f = toDouble(val, p.nu_f);
        else if (key == "nu_d")       p.nu_d = toDouble(val, p.nu_d);
        else if (key == "max_steps")  p.max_steps = toInt(val, p.max_steps);
        else if (key == "max_time")   p.max_time = toDouble(val, p.max_time);

        else if (key == "pcg_state_hi") p.pcg.state_hi = toHex(val, p.pcg.state_hi);
        else if (key == "pcg_state_lo") p.pcg.state_lo = toHex(val, p.pcg.state_lo);
        else if (key == "pcg_inc_hi")   p.pcg.inc_hi = toHex(val, p.pcg.inc_hi);
        else if (key == "pcg_inc_lo")   p.pcg.inc_lo = toHex(val, p.pcg.inc_lo);

        else if (key == "periodic_x")    p.periodic_x = (toInt(val, 1) != 0);
        else if (key == "log_every")     p.log_every = toInt(val, p.log_every);
        else if (key == "snapshot_every") p.snapshot_every = toInt(val, p.snapshot_every);
        else if (key == "save_snapshots") p.save_snapshots = (toInt(val, 1) != 0);
        else if (key == "save_npy")       p.save_npy = (toInt(val, 1) != 0);

        else if (key == "output_dir")    p.output_dir = val;
        else if (key == "history_file")  p.history_filename = val;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Fenwick Tree (Binary Indexed Tree) — mirrors Python FenwickTree class
// ---------------------------------------------------------------------------
class FenwickTree {
public:
    explicit FenwickTree(int size)
        : size_(size), tree_(size + 1, 0.0) {}

    void reset(int size) {
        size_ = size;
        tree_.assign(size + 1, 0.0);
    }

    // Add delta to the element at 0-based index idx.
    void update(int idx, double delta) {
        for (int i = idx + 1; i <= size_; i += i & -i)
            tree_[i] += delta;
    }

    // Return total sum of all elements.
    double total() const {
        int i = size_;
        double s = 0.0;
        while (i > 0) { s += tree_[i]; i -= i & -i; }
        return s;
    }

    // Return smallest 0-based index whose prefix sum >= target.
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
        return idx;  // 0-based
    }

private:
    int size_;
    std::vector<double> tree_;
};

// ---------------------------------------------------------------------------
// Event descriptor (compact, avoids heap allocation per event)
// ---------------------------------------------------------------------------
struct Event {
    bool  is_drop;   // true => drop event; false => hop event
    int16_t sx, sy;  // source site  (valid only for hops)
    int16_t dx, dy;  // destination site raw (before x-wrap)
};

// ---------------------------------------------------------------------------
// History record
// ---------------------------------------------------------------------------
struct HistoryRow {
    std::string label;
    int    step;
    double time;
    int    n_free;
    int    n_deposited;
    int    n_total;
    double total_rate;
};

// ---------------------------------------------------------------------------
// Main simulator class (mirrors ElectrodepositionKMC)
// ---------------------------------------------------------------------------
class ElectrodepositionKMC {
public:
    explicit ElectrodepositionKMC(const KMCParams& p)
        : p_(p),
          rng_(p.pcg),
          lattice_(p.Ny * p.Nx, EMPTY),
          num_drop_(p.Nx),
          num_hop_ (p.Nx * p.Ny * 4),
          max_events_(p.Nx + p.Nx * p.Ny * 4),
          event_rates_(p.Nx + p.Nx * p.Ny * 4, 0.0),
          ftree_(p.Nx + p.Nx * p.Ny * 4),
          idx_to_event_(p.Nx + p.Nx * p.Ny * 4)
    {
        // Validate.
        if (p_.Nx < 1) throw std::invalid_argument("Nx must be >= 1.");
        if (p_.Ny < 2) throw std::invalid_argument("Ny must be >= 2.");
        if (p_.T  <= 0) throw std::invalid_argument("T must be positive.");

        // Substrate row (row 0).
        for (int x = 0; x < p_.Nx; ++x)
            at(x, 0) = SUBSTRATE;

        // Build interaction lookup (indexed by [from_type][to_type]).
        // Matches the Python energy_lookup table exactly.
        memset(energy_lookup_, 0, sizeof(energy_lookup_));
        energy_lookup_[FREE][DEPOSITED]      = p_.e0;
        energy_lookup_[DEPOSITED][FREE]      = p_.e0;
        energy_lookup_[DEPOSITED][DEPOSITED] = p_.e0;
        energy_lookup_[FREE][SUBSTRATE]      = p_.e1;
        energy_lookup_[SUBSTRATE][FREE]      = p_.e1;
        energy_lookup_[DEPOSITED][SUBSTRATE] = p_.e1;
        energy_lookup_[SUBSTRATE][DEPOSITED] = p_.e1;
        energy_lookup_[SUBSTRATE][SUBSTRATE] = p_.e1;

        // Prepare output directory.
        out_dir_ = fs::path(p_.output_dir);
        fs::create_directories(out_dir_);
        if (p_.save_snapshots)
            fs::create_directories(out_dir_ / "snapshots");

        // Build event index table.
        setup_indices();

        // Build initial rate table.
        rebuild_all_rates();

        // Record initial state.
        record_history("initial");
        if (p_.save_snapshots) save_snapshot("initial");
        if (p_.save_npy)       save_lattice_npy("initial");
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    void run_cli() {
        while (step_ < p_.max_steps && time_ < p_.max_time) {
            if (!execute_step()) break;
            if (step_ % p_.log_every == 0)
                record_history("regular");
            if (p_.save_snapshots && step_ % p_.snapshot_every == 0) {
                char tag[32]; snprintf(tag, sizeof(tag), "step_%07d", step_);
                save_snapshot(tag);
            }
            if (p_.save_npy && step_ % p_.snapshot_every == 0) {
                char tag[32]; snprintf(tag, sizeof(tag), "step_%07d", step_);
                save_lattice_npy(tag);
            }
        }
        finalize_outputs();
    }

    int    step() const { return step_; }
    double time() const { return time_; }

private:
    // -----------------------------------------------------------------------
    // Lattice access helpers
    // -----------------------------------------------------------------------
    int8_t& at(int x, int y)       { return lattice_[y * p_.Nx + x]; }
    int8_t  at(int x, int y) const { return lattice_[y * p_.Nx + x]; }

    // Wrap x for periodic BC; returns -1 when out of bounds (non-periodic).
    int wrap_x(int x) const {
        if (p_.periodic_x) return ((x % p_.Nx) + p_.Nx) % p_.Nx;
        if (x >= 0 && x < p_.Nx) return x;
        return -1;
    }

    // Neighbour iteration helper: calls f(nx, ny) for each valid neighbour of (x,y).
    template<typename F>
    void for_each_neighbour(int x, int y, F&& f) const {
        constexpr int dx[4] = {1, -1, 0,  0};
        constexpr int dy[4] = {0,  0, 1, -1};
        for (int i = 0; i < 4; ++i) {
            int nx = wrap_x(x + dx[i]);
            int ny = y + dy[i];
            if (nx == -1 || ny < 0 || ny >= p_.Ny) continue;
            f(nx, ny);
        }
    }

    // -----------------------------------------------------------------------
    // Event indexing (mirrors Python _setup_indices, drop_index, hop_base_index)
    // -----------------------------------------------------------------------
    void setup_indices() {
        int top_y = p_.Ny - 1;
        // Drop events: indices [0, Nx)
        for (int x = 0; x < p_.Nx; ++x) {
            idx_to_event_[x] = { true, 0, 0, (int16_t)x, (int16_t)top_y };
        }
        // Hop events: indices [Nx, Nx + Nx*Ny*4)
        constexpr int DDX[4] = {1, -1, 0,  0};
        constexpr int DDY[4] = {0,  0, 1, -1};
        int base = num_drop_;
        for (int y = 0; y < p_.Ny; ++y) {
            for (int x = 0; x < p_.Nx; ++x) {
                int site_off = (y * p_.Nx + x) * 4;
                for (int d = 0; d < 4; ++d) {
                    int idx = base + site_off + d;
                    idx_to_event_[idx] = {
                        false,
                        (int16_t)x,  (int16_t)y,
                        (int16_t)(x + DDX[d]), (int16_t)(y + DDY[d])
                    };
                }
            }
        }
    }

    int drop_index(int x) const { return x; }
    int hop_base_index(int x, int y) const {
        return num_drop_ + (y * p_.Nx + x) * 4;
    }

    // -----------------------------------------------------------------------
    // Energetics
    // -----------------------------------------------------------------------
    double calc_local_energy(int x, int y, int8_t atom_type) const {
        double e = 0.0;
        for_each_neighbour(x, y, [&](int nx, int ny) {
            e += energy_lookup_[atom_type][(uint8_t)at(nx, ny)];
        });
        return e;
    }

    double get_event_rate(const Event& ev) const {
        if (ev.is_drop) {
            int x1 = ev.dx, y1 = ev.dy;
            return (at(x1, y1) == EMPTY) ? p_.d0 : 0.0;
        }

        int x0 = ev.sx, y0 = ev.sy;
        int8_t atype = at(x0, y0);
        if (atype != FREE && atype != DEPOSITED) return 0.0;

        int x1 = wrap_x(ev.dx);
        int y1 = ev.dy;
        if (x1 == -1 || y1 < 0 || y1 >= p_.Ny) return 0.0;
        if (at(x1, y1) != EMPTY) return 0.0;

        double nu = (atype == FREE) ? p_.nu_f : p_.nu_d;
        double e_init  = calc_local_energy(x0, y0, atype);

        // Temporarily remove atom to compute destination energy.
        const_cast<ElectrodepositionKMC*>(this)->at(x0, y0) = EMPTY;
        double e_final = calc_local_energy(x1, y1, atype);
        const_cast<ElectrodepositionKMC*>(this)->at(x0, y0) = atype;

        return nu * std::exp(-(e_final - e_init) / (2.0 * p_.kB * p_.T));
    }

    void update_rate_at(int idx) {
        double new_rate = get_event_rate(idx_to_event_[idx]);
        double delta    = new_rate - event_rates_[idx];
        if (std::abs(delta) > 1.0e-18) {
            event_rates_[idx] = new_rate;
            ftree_.update(idx, delta);
        }
    }

    void rebuild_all_rates() {
        std::fill(event_rates_.begin(), event_rates_.end(), 0.0);
        ftree_.reset(max_events_);
        for (int i = 0; i < max_events_; ++i)
            update_rate_at(i);
    }

    // -----------------------------------------------------------------------
    // Local rate refresh (radius-2 neighbourhood — mirrors refresh_local_rates)
    // -----------------------------------------------------------------------
    void refresh_local_rates(const std::vector<std::pair<int,int>>& changed) {
        // Collect unique sites within Manhattan distance 2.
        std::vector<std::pair<int,int>> targets;
        targets.reserve(changed.size() * 13);  // ~13 sites per seed

        // Simple dedup via a flat visited set backed by the lattice index.
        std::vector<bool> visited(p_.Nx * p_.Ny, false);

        for (auto [cx, cy] : changed) {
            for (int ddx = -2; ddx <= 2; ++ddx) {
                for (int ddy = -2; ddy <= 2; ++ddy) {
                    if (std::abs(ddx) + std::abs(ddy) > 2) continue;
                    int nx = wrap_x(cx + ddx);
                    int ny = cy + ddy;
                    if (nx == -1 || ny < 0 || ny >= p_.Ny) continue;
                    int linear = ny * p_.Nx + nx;
                    if (!visited[linear]) {
                        visited[linear] = true;
                        targets.emplace_back(nx, ny);
                    }
                }
            }
        }

        int top_y = p_.Ny - 1;
        for (auto [x, y] : targets) {
            if (y == top_y) update_rate_at(drop_index(x));
            int base = hop_base_index(x, y);
            for (int d = 0; d < 4; ++d)
                update_rate_at(base + d);
        }
    }

    // -----------------------------------------------------------------------
    // Bonding-state relaxation (mirrors update_bonding_relaxation)
    // -----------------------------------------------------------------------
    int8_t desired_bond_state(int x, int y) const {
        int8_t st = at(x, y);
        if (st != FREE && st != DEPOSITED) return st;
        bool bonded = false;
        for_each_neighbour(x, y, [&](int nx, int ny) {
            if (at(nx, ny) == DEPOSITED || at(nx, ny) == SUBSTRATE)
                bonded = true;
        });
        return bonded ? DEPOSITED : FREE;
    }

    std::vector<std::pair<int,int>> update_bonding_relaxation(
        const std::vector<std::pair<int,int>>& seeds)
    {
        // BFS queue.
        std::vector<std::pair<int,int>> queue;
        std::vector<bool> in_queue(p_.Nx * p_.Ny, false);
        std::vector<std::pair<int,int>> changed;

        // Seed with each site and its direct neighbours.
        auto enqueue = [&](int x, int y) {
            int lin = y * p_.Nx + x;
            if (!in_queue[lin]) { in_queue[lin] = true; queue.emplace_back(x, y); }
        };
        for (auto [sx, sy] : seeds) {
            enqueue(sx, sy);
            for_each_neighbour(sx, sy, [&](int nx, int ny) { enqueue(nx, ny); });
        }

        size_t head = 0;
        while (head < queue.size()) {
            auto [x, y] = queue[head++];
            int lin = y * p_.Nx + x;
            in_queue[lin] = false;  // allow re-enqueue if needed

            int8_t cur = at(x, y);
            if (cur != FREE && cur != DEPOSITED) continue;

            int8_t desired = desired_bond_state(x, y);
            if (desired == cur) continue;

            at(x, y) = desired;
            changed.emplace_back(x, y);

            // Re-enqueue neighbours and self.
            for_each_neighbour(x, y, [&](int nx, int ny) { enqueue(nx, ny); });
            enqueue(x, y);
        }
        return changed;
    }

    // -----------------------------------------------------------------------
    // KMC step (mirrors execute_step)
    // -----------------------------------------------------------------------
    bool execute_step() {
        double r_tot = ftree_.total();
        if (r_tot <= 0.0) return false;

        // Time increment.
        double u1 = std::max(rng_.next_double(), 1.0e-15);
        double dt  = -std::log(u1) / r_tot;

        // Select event.
        double u2     = std::max(rng_.next_double(), 1.0e-15);
        double target = u2 * r_tot;
        int    idx    = ftree_.find_prefix_index(target);

        const Event& ev = idx_to_event_[idx];
        std::vector<std::pair<int,int>> directly_changed;

        if (ev.is_drop) {
            int x1 = ev.dx, y1 = ev.dy;
            at(x1, y1) = FREE;
            directly_changed.emplace_back(x1, y1);
        } else {
            int x0 = ev.sx, y0 = ev.sy;
            int x1 = wrap_x(ev.dx), y1 = ev.dy;
            int8_t atype = at(x0, y0);
            at(x0, y0) = EMPTY;
            at(x1, y1) = atype;
            directly_changed.emplace_back(x0, y0);
            directly_changed.emplace_back(x1, y1);
        }

        auto relaxed = update_bonding_relaxation(directly_changed);

        // Merge changed sets.
        std::vector<std::pair<int,int>> all_changed = directly_changed;
        all_changed.insert(all_changed.end(), relaxed.begin(), relaxed.end());
        refresh_local_rates(all_changed);

        time_ += dt;
        ++step_;
        return true;
    }

    // -----------------------------------------------------------------------
    // Output helpers
    // -----------------------------------------------------------------------
    struct Counts { int free, dep, total; };
    Counts counts() const {
        int nf = 0, nd = 0;
        for (auto v : lattice_) {
            if (v == FREE)      ++nf;
            else if (v == DEPOSITED) ++nd;
        }
        return {nf, nd, nf + nd};
    }

    void record_history(const std::string& label) {
        auto [nf, nd, nt] = counts();
        double tr = 0.0;
        for (double r : event_rates_) tr += r;
        history_.push_back({label, step_, time_, nf, nd, nt, tr});
    }

    void write_history_csv() const {
        fs::path out = out_dir_ / p_.history_filename;
        std::ofstream f(out);
        if (!f) throw std::runtime_error("Cannot write history CSV: " + out.string());
        f << "label,step,time,free,deposited,total_mobile_plus_deposited,total_rate\n";
        for (const auto& row : history_) {
            f << row.label    << ','
              << row.step     << ','
              << std::scientific << std::setprecision(6) << row.time << ','
              << row.n_free   << ','
              << row.n_deposited << ','
              << row.n_total  << ','
              << row.total_rate << '\n';
        }
    }

    // Write a colour PPM (P6) snapshot scaled up so each lattice cell is CELL_PX pixels.
    // PPM is supported by most image viewers, GIMP, Photoshop, and IrfanView without plugins.
    // Colors: black=EMPTY  steel-blue=FREE  amber=DEPOSITED  dark-grey=SUBSTRATE
    void save_snapshot(const std::string& tag) const {
        const int CELL_PX = std::max(8, std::min(24, 400 / std::max(p_.Nx, p_.Ny)));
        const int IMG_W   = p_.Nx * CELL_PX;
        const int IMG_H   = p_.Ny * CELL_PX;

        fs::path out = out_dir_ / "snapshots" / (tag + ".ppm");
        std::ofstream f(out, std::ios::binary);
        if (!f) { std::cerr << "Warning: cannot write snapshot " << out << '\n'; return; }

        // PPM header with metadata comment.
        f << "P6\n"
          << "# LKMC | " << tag
          << " | step=" << step_
          << " | time=" << std::scientific << std::setprecision(3) << time_
          << " | T=" << p_.T << "K | Nx=" << p_.Nx << " Ny=" << p_.Ny << "\n"
          << "# Colors: black=empty  blue=free  orange=deposited  darkgrey=substrate\n"
          << IMG_W << ' ' << IMG_H << "\n255\n";

        struct RGB { uint8_t r, g, b; };
        static const RGB PAL[4] = {
            {0x11, 0x11, 0x11},   // EMPTY
            {0x55, 0x99, 0xdd},   // FREE      (steel blue)
            {0xdd, 0x88, 0x33},   // DEPOSITED (amber)
            {0x22, 0x22, 0x22},   // SUBSTRATE (dark grey)
        };

        // Write rows top-to-bottom (lattice row 0 = substrate = bottom of image).
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

    // Write the raw lattice as a simple binary file: 4-byte header (Ny, Nx),
    // then Ny*Nx int8 values in row-major order (row 0 = substrate).
    void save_lattice_npy(const std::string& tag) const {
        fs::path out = out_dir_ / ("lattice_" + tag + ".bin");
        std::ofstream f(out, std::ios::binary);
        if (!f) { std::cerr << "Warning: cannot write lattice bin " << out << '\n'; return; }
        int32_t header[2] = {(int32_t)p_.Ny, (int32_t)p_.Nx};
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
        f.write(reinterpret_cast<const char*>(lattice_.data()), lattice_.size());
    }

    void finalize_outputs() {
        record_history("final");
        std::string tag = "final_step_" + std::to_string(step_);
        if (p_.save_snapshots) save_snapshot(tag);
        if (p_.save_npy)       save_lattice_npy(tag);
        write_history_csv();
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------
    KMCParams p_;
    PCG64 rng_; 

    std::vector<int8_t>  lattice_;       // [y*Nx + x]
    double energy_lookup_[4][4];

    int num_drop_;
    int num_hop_;
    int max_events_;

    std::vector<double> event_rates_;
    FenwickTree         ftree_;
    std::vector<Event>  idx_to_event_;

    double time_ = 0.0;
    int    step_ = 0;

    fs::path out_dir_;
    std::vector<HistoryRow> history_;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
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

    std::cout << "LKMC Electrodeposition  (C++ port)\n"
              << "  Lattice : " << params.Nx << " x " << params.Ny << '\n'
              << "  T       : " << params.T  << " K\n"
              << "  d0      : " << params.d0 << "  e0 : " << params.e0
              << "  e1 : " << params.e1 << '\n'
              << "  max_steps : " << params.max_steps
              << "  max_time : " << params.max_time << " s\n"
              << "  pcg state: " << std::hex << params.pcg.state_hi << "_" << params.pcg.state_lo << std::dec << '\n'
              << std::flush;

    auto t0 = std::chrono::steady_clock::now();

    try {
        ElectrodepositionKMC sim(params);
        sim.run_cli();

        auto t1   = std::chrono::steady_clock::now();
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