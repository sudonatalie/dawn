RWByteAddressBuffer prevent_dce : register(u0);

vector<float16_t, 2> quadBroadcast_3c3824() {
  vector<float16_t, 2> res = QuadReadLaneAt((float16_t(1.0h)).xx, 1u);
  return res;
}

void fragment_main() {
  prevent_dce.Store<vector<float16_t, 2> >(0u, quadBroadcast_3c3824());
  return;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store<vector<float16_t, 2> >(0u, quadBroadcast_3c3824());
  return;
}
