SKIP: FAILED


static vector<float16_t, 3> u = (float16_t(1.0h)).xxx;
[numthreads(1, 1, 1)]
void unused_entry_point() {
}

FXC validation failure:
c:\src\dawn\Shader@0x0000023261D8D0A0(2,15-23): error X3000: syntax error: unexpected token 'float16_t'

