#pragma once

#include <cstdint>

namespace sgpu {

constexpr char kPsiQrSnapshotMagic[8] = {'S', 'G', 'P', 'U', 'Q', 'R', '1', '\0'};
constexpr std::uint32_t kPsiQrSnapshotVersion = 1;
constexpr std::uint32_t kPsiQrScalarFloat32 = 1;
constexpr std::uint32_t kPsiQrLayoutColumnMajor = 1;

#pragma pack(push, 1)
struct PsiQrSnapshotHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t scalar_type;
    std::uint32_t layout;
    std::uint32_t reserved;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t data_rows;
    double ridge;
    std::uint64_t matrix_bytes;
    std::uint64_t rhs_bytes;
    std::uint64_t scale_bytes;
};
#pragma pack(pop)

static_assert(sizeof(PsiQrSnapshotHeader) == 80, "unexpected psi QR snapshot header size");

}  // namespace sgpu
