enable subgroups;
enable subgroups_f16;
enable f16;

@group(0) @binding(0) var<storage, read_write> prevent_dce : f16;

fn subgroupExclusiveMul_6f431e() -> f16 {
  var arg_0 = 1.0h;
  var res : f16 = subgroupExclusiveMul(arg_0);
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupExclusiveMul_6f431e();
}
