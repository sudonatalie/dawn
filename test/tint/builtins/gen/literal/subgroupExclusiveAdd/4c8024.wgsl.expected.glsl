SKIP: FAILED


@group(0) @binding(0) var<storage, read_write> prevent_dce : vec2<f32>;

fn subgroupExclusiveAdd_4c8024() -> vec2<f32> {
  var res : vec2<f32> = subgroupExclusiveAdd(vec2<f32>(1.0f));
  return res;
}

@compute @workgroup_size(1)
fn compute_main() {
  prevent_dce = subgroupExclusiveAdd_4c8024();
}

Failed to generate: error: Unknown builtin method: 0x55fe91648230
