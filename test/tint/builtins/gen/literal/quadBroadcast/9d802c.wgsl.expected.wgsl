enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f32>;

fn quadBroadcast_9d802c() -> vec4<f32> {
  var res : vec4<f32> = quadBroadcast(vec4<f32>(1.0f), 1i);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = quadBroadcast_9d802c();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = quadBroadcast_9d802c();
}
