SKIP: FAILED


static float16_t t = float16_t(0.0h);
vector<float16_t, 4> m() {
  t = float16_t(1.0h);
  return vector<float16_t, 4>((t).xxxx);
}

void f() {
  bool4 v = bool4(m());
}

[numthreads(1, 1, 1)]
void unused_entry_point() {
}

FXC validation failure:
c:\src\dawn\Shader@0x00000143FAE90AE0(2,8-16): error X3000: unrecognized identifier 'float16_t'

