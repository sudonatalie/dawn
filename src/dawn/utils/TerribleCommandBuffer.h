// Copyright 2017 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SRC_DAWN_UTILS_TERRIBLECOMMANDBUFFER_H_
#define SRC_DAWN_UTILS_TERRIBLECOMMANDBUFFER_H_

#include "dawn/wire/Wire.h"
#include "partition_alloc/pointers/raw_ptr.h"

namespace dawn::utils {

class TerribleCommandBuffer : public dawn::wire::CommandSerializer {
  public:
    TerribleCommandBuffer();
    explicit TerribleCommandBuffer(dawn::wire::CommandHandler* handler);

    void SetHandler(dawn::wire::CommandHandler* handler);

    size_t GetMaximumAllocationSize() const override;

    void* GetCmdSpace(size_t size) override;
    bool Flush() override;

  private:
    // TODO(https://crbug/dawn/2343): Remove DanglingUntriaged.
    raw_ptr<dawn::wire::CommandHandler, DanglingUntriaged> mHandler = nullptr;
    size_t mOffset = 0;
    char mBuffer[1000000];
};

}  // namespace dawn::utils

#endif  // SRC_DAWN_UTILS_TERRIBLECOMMANDBUFFER_H_
