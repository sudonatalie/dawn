RWByteAddressBuffer prevent_dce : register(u0);

float3 quadBroadcast_0cc513() {
  float3 arg_0 = (1.0f).xxx;
  float3 res = QuadReadLaneAt(arg_0, 1u);
  return res;
}

void fragment_main() {
  prevent_dce.Store3(0u, asuint(quadBroadcast_0cc513()));
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store3(0u, asuint(quadBroadcast_0cc513()));
  return;
}
