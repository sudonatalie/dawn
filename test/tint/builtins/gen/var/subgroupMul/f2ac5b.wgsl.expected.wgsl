enable subgroups;
enable subgroups_f16;
enable f16;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f16>;

fn subgroupMul_f2ac5b() -> vec4<f16> {
  var arg_0 = vec4<f16>(1.0h);
  var res : vec4<f16> = subgroupMul(arg_0);
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupMul_f2ac5b();
}
