# Plan: Fix Cross-Compilation Build for Dawn

## Current Status
The Dawn release workflow was failing due to a Protocol Buffer generation error during cross-compilation (Android and iOS). This was caused by the build system attempting to run a target-architecture `protoc` binary on the host machine.

A minimal fix has been applied to `.github/workflows/ci.yml` to rely on the host environment's `protoc` and pass its path to CMake.

## Background Information
- **Failure Window:** Started around April 25, 2026.
- **Root Cause:** Upstream Protocol Buffers (via Chromium) removed pre-generated C++ files for "Well Known Types" (WKT), forcing generation during the build.
- **Trigger:** Cross-compilation environments (Android NDK, iOS toolchain) build `protoc` for the *target*, but CMake tries to execute it on the *host* to generate WKT code.

## Strategy: Minimal CI Fix
The host `protoc` is assumed to be available in the GitHub Actions environment (or installed in the container) and is passed to the build system.

### Tasks
1.  **Update `ci.yml`**: 
    *   Added `protobuf-compiler` to the `manylinux` container dependencies.
    *   Pass `-DPROTOC_EXECUTABLE=$(which protoc)` to CMake in the mobile-android and mobile-apple jobs, leveraging pre-installed tools on `ubuntu-latest` and `macos-latest`.
2.  **Verify**: Monitor GitHub Actions to see if the Android and iOS builds succeed with this minimal change.

## Next Steps
- If this fixes Android/iOS, consider if similar minimal changes are needed elsewhere.
- Evaluate if a more robust solution is required if environment-provided `protoc` versions diverge too much from Dawn's requirements.
