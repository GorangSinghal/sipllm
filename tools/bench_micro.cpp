#include "llm/auto_tuner.h"
#include "llm/device_profile.h"
#include <cstdio>
#include <string>

using namespace llm;

int main(int argc, char** argv) {
    bool recalibrate = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--recalibrate") {
            recalibrate = true;
        }
    }

    HardwareInfo hw = get_hardware_info();
    printf("Hardware ID: %s\n", hw.hardware_id().c_str());
    printf("%s\n", hw.to_json(2).c_str());

    AutoTunerOptions opt;
    opt.force_recalibrate = recalibrate;
    
    RuntimeProfile profile = tune_if_needed(hw, opt);
    printf("Optimal Profile:\n%s\n", profile.to_json(2).c_str());
    
    return 0;
}
