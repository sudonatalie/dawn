enable subgroups;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<i32>;

fn subgroupExclusiveMul_4525a3() -> vec2<i32> {
  var res : vec2<i32> = subgroupExclusiveMul(vec2<i32>(1i));
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupExclusiveMul_4525a3();
}
