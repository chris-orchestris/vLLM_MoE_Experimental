// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

constexpr int kBarrierCount = 33;

struct alignas(64) BarrierState {
  std::atomic<std::uint32_t> arrivals;
  std::atomic<std::uint32_t> generation;
  std::byte padding[64 - 2 * sizeof(std::atomic<std::uint32_t>)];
};

struct SharedHeader {
  BarrierState barriers[kBarrierCount];
};

static_assert(sizeof(BarrierState) == 64);
static_assert(sizeof(SharedHeader) <= 4096);

}  // namespace

extern "C" void vllm_host_barriers_init(void* header) {
  auto* shared = static_cast<SharedHeader*>(header);
  for (auto& barrier : shared->barriers) {
    barrier.arrivals.store(0, std::memory_order_relaxed);
    barrier.generation.store(0, std::memory_order_relaxed);
  }
}

extern "C" void vllm_host_barrier(
    void* header, int barrier_index, std::uint32_t participants) {
  if (barrier_index < 0 || barrier_index >= kBarrierCount) {
    return;
  }
  auto& barrier = static_cast<SharedHeader*>(header)->barriers[barrier_index];
  const std::uint32_t generation =
      barrier.generation.load(std::memory_order_acquire);
  const std::uint32_t arrival =
      barrier.arrivals.fetch_add(1, std::memory_order_acq_rel);
  if (arrival + 1 == participants) {
    barrier.arrivals.store(0, std::memory_order_relaxed);
    barrier.generation.fetch_add(1, std::memory_order_release);
    return;
  }
  while (barrier.generation.load(std::memory_order_acquire) == generation) {
#if defined(__x86_64__)
    _mm_pause();
#endif
  }
}
