#pragma once

#include "llm/device_profile.h"
#include <string>

namespace llm {

struct AutoTunerOptions {
    bool force_recalibrate = false;
    bool disable_autotune = false;
};

// Top-level entrypoint: loads the profile if it exists, otherwise runs
// the micro-benchmarks to generate it.
RuntimeProfile tune_if_needed(const HardwareInfo& hw, const AutoTunerOptions& opt);

// Manual runner for the bench_micro CLI tool
RuntimeProfile run_micro_benchmarks(const HardwareInfo& hw);

} // namespace llm
