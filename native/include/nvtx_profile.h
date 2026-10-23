#pragma once

#ifdef SGPU_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

class SgpuNvtxRange {
public:
    explicit SgpuNvtxRange(const char* name) {
#ifdef SGPU_ENABLE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~SgpuNvtxRange() {
#ifdef SGPU_ENABLE_NVTX
        nvtxRangePop();
#endif
    }

    SgpuNvtxRange(const SgpuNvtxRange&) = delete;
    SgpuNvtxRange& operator=(const SgpuNvtxRange&) = delete;
};

#define SGPU_NVTX_CONCAT_INNER(a, b) a##b
#define SGPU_NVTX_CONCAT(a, b) SGPU_NVTX_CONCAT_INNER(a, b)
#define SGPU_NVTX_RANGE(name) \
    SgpuNvtxRange SGPU_NVTX_CONCAT(sgpu_nvtx_range_, __LINE__)(name)
