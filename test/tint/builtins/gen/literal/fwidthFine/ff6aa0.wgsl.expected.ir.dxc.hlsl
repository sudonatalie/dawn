SKIP: FAILED

float2 fwidthFine_ff6aa0() {
  float2 res = fwidth((1.0f).xx);
  return res;
}

void fragment_main() {
  prevent_dce = fwidthFine_ff6aa0();
}

DXC validation failure:
hlsl.hlsl:7:3: error: use of undeclared identifier 'prevent_dce'
  prevent_dce = fwidthFine_ff6aa0();
  ^

