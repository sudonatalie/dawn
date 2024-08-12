enable subgroups;
enable subgroups_f16;
enable f16;

@group(0) @binding(0) var<storage, read_write> prevent_dce : vec4<f16>;

fn subgroupShuffleUp_0990cd() -> vec4<f16> {
  var res : vec4<f16> = subgroupShuffleUp(vec4<f16>(1.0h), 1u);
  return res;
}

@fragment
fn fragment_main() {
  prevent_dce = subgroupShuffleUp_0990cd();
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupShuffleUp_0990cd();
}
