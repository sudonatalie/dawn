SKIP: Wave ops not support before SM6.0

RWByteAddressBuffer prevent_dce : register(u0);

uint3 subgroupBroadcastFirst_5e5b6f() {
  uint3 res = WaveReadLaneFirst((1u).xxx);
  return res;
}

void fragment_main() {
  prevent_dce.Store3(0u, asuint(subgroupBroadcastFirst_5e5b6f()));
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store3(0u, asuint(subgroupBroadcastFirst_5e5b6f()));
  return;
}
