# Copyright 2023 The Tint Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# File generated by 'tools/src/cmd/gen' using the template:
#   tools/src/cmd/gen/build/BUILD.cmake.tmpl
#
# To regenerate run: './tools/run gen'
#
#                       Do not modify this file directly
################################################################################

include(lang/spirv/intrinsic/BUILD.cmake)
include(lang/spirv/ir/BUILD.cmake)
include(lang/spirv/reader/BUILD.cmake)
include(lang/spirv/type/BUILD.cmake)
include(lang/spirv/writer/BUILD.cmake)

################################################################################
# Target:    tint_lang_spirv
# Kind:      lib
################################################################################
tint_add_target(tint_lang_spirv lib
  lang/spirv/builtin_fn.cc
  lang/spirv/builtin_fn.h
)

tint_target_add_dependencies(tint_lang_spirv lib
  tint_utils_traits
)
