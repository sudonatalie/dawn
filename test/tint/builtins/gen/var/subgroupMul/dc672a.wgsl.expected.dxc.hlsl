RWByteAddressBuffer prevent_dce : register(u0);

uint2 subgroupMul_dc672a() {
  uint2 arg_0 = (1u).xx;
  uint2 res = WaveActiveProduct(arg_0);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store2(0u, asuint(subgroupMul_dc672a()));
  return;
}
