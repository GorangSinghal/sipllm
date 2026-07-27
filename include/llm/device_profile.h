#pragma once

#include <string>
#include <cstddef>
#include <cstdint>

namespace llm {

// Static hardware traits collected once per device
struct HardwareInfo {
    std::string cpu_model;
    std::string architecture;
    int         logical_cores = 0;
    int         physical_cores = 0;
    size_t      ram_bytes = 0;
    size_t      page_size = 0;
    
    // Deterministic hash based on the traits above, used as part of profile keys
    std::string hardware_id() const;
    
    // Convert to JSON for persistence
    std::string to_json(int indent = 0) const;
};

// Auto-detected or manually overridden runtime tunables
struct RuntimeProfile {
    int threads = 4;
    int schedule_policy = 0;
    
    // Performance stats measured during tuning
    double thread_barrier_ms = 0;
    double thread_throughput_relative = 0;
    
    std::string measurements_json = "{}"; // raw string for JSON insertion
    
    std::string to_json(int indent = 0) const;
};

// Retrieve hardware traits (reads from OS)
HardwareInfo get_hardware_info();

// Fetch paths for the persistence layer
std::string get_sipllm_home();
std::string get_hardware_profile_path();
std::string get_runtime_profile_path(const std::string& hw_id);

// Load saved profiles from disk (returns true if found and parsed)
bool load_hardware_profile(HardwareInfo& out);
bool load_runtime_profile(const std::string& hw_id, RuntimeProfile& out);

// Save profiles to disk
void save_hardware_profile(const HardwareInfo& info);
void save_runtime_profile(const std::string& hw_id, const RuntimeProfile& profile);

} // namespace llm
