#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

namespace gpt {

// ─────────────────────────────────────────────────────────────────────────────
//  ScopedTimer — RAII timer, prints on destruction
// ─────────────────────────────────────────────────────────────────────────────
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), t0_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto dt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0_).count();
        std::cout << "[Timer] " << name_ << ": " << std::fixed
                  << std::setprecision(2) << dt << " ms\n";
    }
private:
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Profiler — accumulates named timings across multiple calls
// ─────────────────────────────────────────────────────────────────────────────
class Profiler {
public:
    static Profiler& get() { static Profiler p; return p; }

    void start(const std::string& name) {
        starts_[name] = std::chrono::steady_clock::now();
    }

    void stop(const std::string& name) {
        auto it = starts_.find(name);
        if (it == starts_.end()) return;
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - it->second).count();
        records_[name].push_back(ms);
    }

    void report() const {
        std::cout << "\n=== Profiler Report ===\n";
        std::cout << std::left << std::setw(30) << "Name"
                  << std::right << std::setw(8) << "Calls"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Mean(ms)"
                  << std::setw(12) << "Min(ms)"
                  << std::setw(12) << "Max(ms)" << "\n";
        std::cout << std::string(86, '-') << "\n";
        for (auto& [name, vals] : records_) {
            double total = std::accumulate(vals.begin(), vals.end(), 0.0);
            double mean  = total / vals.size();
            double mn    = *std::min_element(vals.begin(), vals.end());
            double mx    = *std::max_element(vals.begin(), vals.end());
            std::cout << std::left  << std::setw(30) << name
                      << std::right << std::setw(8)  << vals.size()
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << total
                      << std::setw(12) << mean
                      << std::setw(12) << mn
                      << std::setw(12) << mx << "\n";
        }
        std::cout << "======================\n\n";
    }

    void reset() { records_.clear(); starts_.clear(); }

private:
    std::unordered_map<std::string, std::vector<double>> records_;
    std::unordered_map<std::string,
        std::chrono::steady_clock::time_point>           starts_;
};

// RAII profiler scope
struct ProfileScope {
    std::string name;
    ProfileScope(const std::string& n) : name(n) { Profiler::get().start(n); }
    ~ProfileScope() { Profiler::get().stop(name); }
};

#define PROFILE_SCOPE(name) gpt::ProfileScope _ps_##__LINE__(name)

} // namespace gpt
