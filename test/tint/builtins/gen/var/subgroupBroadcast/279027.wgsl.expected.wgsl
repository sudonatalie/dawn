enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<u32>;

fn subgroupBroadcast_279027() -> vec4<u32> {
  var arg_0 = vec4<u32>(1u);
  const arg_1 = 1u;
  var res : vec4<u32> = subgroupBroadcast(arg_0, arg_1);
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupBroadcast_279027();
}
