SKIP: FAILED

float3 dpdxCoarse_f64d7b() {
  float3 arg_0 = (1.0f).xxx;
  float3 res = ddx_coarse(arg_0);
  return res;
}

void fragment_main() {
  prevent_dce = dpdxCoarse_f64d7b();
}

