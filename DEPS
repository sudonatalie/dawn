use_relative_paths = True
use_relative_hooks = True

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'dawn_git': 'https://dawn.googlesource.com',
  'github_git': 'https://github.com',
  'swiftshader_git': 'https://swiftshader.googlesource.com',

  'dawn_standalone': True,
}

deps = {
  # Dependencies required to use GN/Clang in standalone
  'build': {
    'url': '{chromium_git}/chromium/src/build@896323eeda1bd1b01156b70625d5e14de225ebc3',
    'condition': 'dawn_standalone',
  },
  'buildtools': {
    'url': '{chromium_git}/chromium/src/buildtools@2c41dfb19abe40908834803b6fed797b0f341fe1',
    'condition': 'dawn_standalone',
  },
  'tools/clang': {
    'url': '{chromium_git}/chromium/src/tools/clang@698732d5db36040c07d5cc5f9137fcc943494c11',
    'condition': 'dawn_standalone',
  },
  'third_party/binutils': {
    'url': '{chromium_git}/chromium/src/third_party/binutils@f9ce777698a819dff4d6a033b31122d91a49b62e',
    'condition': 'dawn_standalone',
  },
  'tools/clang/dsymutil': {
    'packages': [
      {
        'package': 'chromium/llvm-build-tools/dsymutil',
        'version': 'M56jPzDv1620Rnm__jTMYS62Zi8rxHVq7yw0qeBFEgkC',
      }
    ],
    'condition': 'checkout_mac or checkout_ios',
    'dep_type': 'cipd',
  },

  # Testing, GTest and GMock
  'testing': {
    'url': '{chromium_git}/chromium/src/testing@e5ced5141379ee8ae28b4f93d3c02df039d2b052',
    'condition': 'dawn_standalone',
  },
  'third_party/googletest': {
    'url': '{chromium_git}/external/github.com/google/googletest@a09ea700d32bab83325aff9ff34d0582e50e3997',
    'condition': 'dawn_standalone',
  },

  # Jinja2 and MarkupSafe for the code generator
  'third_party/jinja2': {
    'url': '{chromium_git}/chromium/src/third_party/jinja2@b41863e42637544c2941b574c7877d3e1f663e25',
    'condition': 'dawn_standalone',
  },
  'third_party/markupsafe': {
    'url': '{chromium_git}/chromium/src/third_party/markupsafe@8f45f5cfa0009d2a70589bcda0349b8cb2b72783',
    'condition': 'dawn_standalone',
  },

  # SPIRV-Cross
  'third_party/spirv-cross': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/SPIRV-Cross@559b21c6c91f65ba52cdfa7a76e1185cd3c8f144',
    'condition': 'dawn_standalone',
  },

  # SPIRV compiler dependencies: SPIRV-Tools, SPIRV-headers, glslang and shaderc
  'third_party/SPIRV-Tools': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/SPIRV-Tools@2fbeb04b6e6b6da58fd5fe7f801a9de5b32f656f',
    'condition': 'dawn_standalone',
  },
  'third_party/spirv-headers': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/SPIRV-Headers@308bd07424350a6000f35a77b5f85cd4f3da319e',
    'condition': 'dawn_standalone',
  },
  'third_party/glslang': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/glslang@29c9135d06f12628e3161e4aa751610c5941d8d6',
    'condition': 'dawn_standalone',
  },
  'third_party/shaderc': {
    'url': '{chromium_git}/external/github.com/google/shaderc@1402ed576596f53eaf3f8d390a9e1defbeffd014',
    'condition': 'dawn_standalone',
  },

  # WGSL support
  'third_party/tint': {
    'url': '{dawn_git}/tint@7b819aa162403eceb22133de44c15f20e3bcd862',
    'condition': 'dawn_standalone',
  },

  # GLFW for tests and samples
  'third_party/glfw': {
    'url': '{chromium_git}/external/github.com/glfw/glfw@d973acc123826666ecc9e6fd475682e3d84c54a6',
    'condition': 'dawn_standalone',
  },

  # Dependencies for samples: GLM
  'third_party/glm': {
    'url': '{github_git}/g-truc/glm.git@bf71a834948186f4097caa076cd2663c69a10e1e',
    'condition': 'dawn_standalone',
  },

  # Khronos Vulkan headers, validation layers and loader.
  'third_party/vulkan-headers': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/Vulkan-Headers@4c079bf40c2587220dbf157d825d3185c9adc896',
    'condition': 'dawn_standalone',
  },
  'third_party/vulkan-validation-layers': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/Vulkan-ValidationLayers@e8b96e86fe2edfaee274b98fbbe1bd65579b0904',
    'condition': 'dawn_standalone',
  },
  'third_party/vulkan-loader': {
    'url': '{chromium_git}/external/github.com/KhronosGroup/Vulkan-Loader@006586926adece57adea3e006140b5df19826371',
    'condition': 'dawn_standalone',
  },

  'third_party/swiftshader': {
    'url': '{swiftshader_git}/SwiftShader@f6f11215cd930f4609bc8714178ef0a4ec3ca26e',
    'condition': 'dawn_standalone',
  },

}

hooks = [
  # Pull clang-format binaries using checked-in hashes.
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'condition': 'host_os == "win" and dawn_standalone',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'condition': 'host_os == "mac" and dawn_standalone',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'condition': 'host_os == "linux" and dawn_standalone',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'buildtools/linux64/clang-format.sha1',
    ],
  },

  # Pull the compilers and system libraries for hermetic builds
  {
    'name': 'sysroot_x86',
    'pattern': '.',
    'condition': 'checkout_linux and ((checkout_x86 or checkout_x64) and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and (checkout_x64 and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Windows toolchain if necessary. Must run before 'clang' below.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win and dawn_standalone',
    'action': ['python', 'build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Note: On Win, this should run after win_toolchain, as it may use it.
    'name': 'clang',
    'pattern': '.',
    'action': ['python', 'tools/clang/scripts/update.py'],
    'condition': 'dawn_standalone',
  },
  {
    # Pull rc binaries using checked-in hashes.
    'name': 'rc_win',
    'pattern': '.',
    'condition': 'checkout_win and (host_os == "win" and dawn_standalone)',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-browser-clang/rc',
                '-s', 'build/toolchain/win/rc/win/rc.exe.sha1',
    ],
  },
  # Pull binutils for linux hermetic builds
  {
    'name': 'binutils',
    'pattern': 'src/third_party/binutils',
    'condition': 'host_os == "linux" and dawn_standalone',
    'action': [
        'python',
        'third_party/binutils/download.py',
    ],
  },
  # Update build/util/LASTCHANGE.
  {
    'name': 'lastchange',
    'pattern': '.',
    'condition': 'dawn_standalone',
    'action': ['python', 'build/util/lastchange.py',
               '-o', 'build/util/LASTCHANGE'],
  },
]

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++abi
  'buildtools',
]
