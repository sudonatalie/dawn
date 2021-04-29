// Copyright 2021 The Tint Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fuzzers/tint_common_fuzzer.h"

namespace tint {
namespace fuzzers {

struct Config {
  const uint8_t* data;
  size_t size;
  transform::Manager manager;
  transform::DataMap inputs;
};

bool AddPlatformIndependentPasses(Config* config) {
  if (!ExtractFirstIndexOffsetInputs(&config->data, &config->size,
                                     &config->inputs)) {
    return false;
  }

  if (!ExtractBindingRemapperInputs(&config->data, &config->size,
                                    &config->inputs)) {
    return false;
  }

  config->manager.Add<transform::BoundArrayAccessors>();
  config->manager
      .Add<transform::EmitVertexPointSize>();  // TODO(tint:753): Remove once
                                               // transform used by sanitizers
  config->manager.Add<transform::FirstIndexOffset>();
  config->manager.Add<transform::BindingRemapper>();

  return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  {
    Config config;
    config.data = data;
    config.size = size;

    if (!AddPlatformIndependentPasses(&config)) {
      return 0;
    }

    fuzzers::CommonFuzzer fuzzer(InputFormat::kWGSL, OutputFormat::kSpv);
    fuzzer.SetTransformManager(&(config.manager), std::move(config.inputs));

    fuzzer.Run(config.data, config.size);
  }

#if TINT_BUILD_HLSL_WRITER
  {
    Config config;
    config.data = data;
    config.size = size;

    if (!AddPlatformIndependentPasses(&config)) {
      return 0;
    }

    config.manager.Add<transform::Hlsl>();

    fuzzers::CommonFuzzer fuzzer(InputFormat::kWGSL, OutputFormat::kHLSL);
    fuzzer.SetTransformManager(&config.manager, std::move(config.inputs));

    fuzzer.Run(config.data, config.size);
  }
#endif  // TINT_BUILD_HLSL_WRITER

#if TINT_BUILD_MSL_WRITER
  {
    Config config;
    config.data = data;
    config.size = size;

    if (!AddPlatformIndependentPasses(&config)) {
      return 0;
    }

    config.manager.Add<transform::Msl>();

    fuzzers::CommonFuzzer fuzzer(InputFormat::kWGSL, OutputFormat::kMSL);
    fuzzer.SetTransformManager(&config.manager, std::move(config.inputs));

    fuzzer.Run(config.data, config.size);
  }
#endif  // TINT_BUILD_MSL_WRITER

  return 0;
}

}  // namespace fuzzers
}  // namespace tint
