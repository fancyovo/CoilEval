#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Opt-in diagnostic wall spans. Disabled calls do not read the clock or emit IO.
class CoilEvalHostProfile {
    const char* stage_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
public:
    explicit CoilEvalHostProfile(const char* stage):stage_(stage) {
        const char* flag=std::getenv("COIL_EVAL_PROFILE");
        enabled_=flag && flag[0]=='1';
        if(enabled_) start_=std::chrono::steady_clock::now();
    }
    ~CoilEvalHostProfile() {
        if(enabled_) {
            const double seconds=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-start_).count();
            std::fprintf(stderr,"COIL_EVAL_PROFILE {\"stage\":\"%s\",\"seconds\":%.9f}\n",stage_,seconds);
        }
    }
};
