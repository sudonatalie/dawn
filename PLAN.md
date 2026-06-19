# Plan: Fix Cross-Compilation Build for Dawn

## Current Status
The Dawn release workflow was failing due to a Protocol Buffer generation error during cross-compilation (Android and iOS). This was caused by the build system attempting to run a target-architecture `protoc` binary on the host machine.

A minimal and isolated fix has been applied to `.github/workflows/ci.yml`.

## Background Information
- **Failure Window:** Started around April 25, 2026.
- **Root Cause:** Upstream Protocol Buffers (via Chromium) removed pre-generated C++ files for "Well Known Types" (WKT), forcing generation during the build.
- **Trigger:** Cross-compilation environments (Android NDK, iOS toolchain) build `protoc` for the *target*, but CMake tries to execute it on the *host* to generate WKT code.

## Strategy: Isolated CI Fix
The host `protoc` is provided by the GitHub Actions environment specifically for cross-compilation jobs.

### Completed Tasks
1.  **Update `ci.yml`**: 
    *   **mobile-android**: Added a step to install `protobuf-compiler` and passed `-DPROTOC_EXECUTABLE=$(which protoc)`.
    *   **mobile-apple**: Added a step to install `protobuf` and passed `-DPROTOC_EXECUTABLE=$(which protoc)`.
2.  **Verify**: Reverted changes to `third_party/protobuf.cmake` and `package-emdawnwebgpu.sh` to isolate the fix to the mobile release workflows in `ci.yml`.

## Next Steps
- Monitor GitHub Actions to confirm the Android and iOS builds succeed.
- If successful, consider applying a similar minimal fix to `package-emdawnwebgpu.sh` for WASM.
