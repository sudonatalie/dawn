// Copyright 2023 The Dawn & Tint Authors
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

////////////////////////////////////////////////////////////////////////////////
// File generated by 'tools/src/cmd/gen' using the template:
//   src/tint/lang/core/builtin_fn.cc.tmpl
//
// To regenerate run: './tools/run gen'
//
//                       Do not modify this file directly
////////////////////////////////////////////////////////////////////////////////

#include "src/tint/lang/core/builtin_fn.h"

namespace tint::core {

BuiltinFn ParseBuiltinFn(std::string_view name) {
    if (name == "abs") {
        return BuiltinFn::kAbs;
    }
    if (name == "acos") {
        return BuiltinFn::kAcos;
    }
    if (name == "acosh") {
        return BuiltinFn::kAcosh;
    }
    if (name == "all") {
        return BuiltinFn::kAll;
    }
    if (name == "any") {
        return BuiltinFn::kAny;
    }
    if (name == "arrayLength") {
        return BuiltinFn::kArrayLength;
    }
    if (name == "asin") {
        return BuiltinFn::kAsin;
    }
    if (name == "asinh") {
        return BuiltinFn::kAsinh;
    }
    if (name == "atan") {
        return BuiltinFn::kAtan;
    }
    if (name == "atan2") {
        return BuiltinFn::kAtan2;
    }
    if (name == "atanh") {
        return BuiltinFn::kAtanh;
    }
    if (name == "ceil") {
        return BuiltinFn::kCeil;
    }
    if (name == "clamp") {
        return BuiltinFn::kClamp;
    }
    if (name == "cos") {
        return BuiltinFn::kCos;
    }
    if (name == "cosh") {
        return BuiltinFn::kCosh;
    }
    if (name == "countLeadingZeros") {
        return BuiltinFn::kCountLeadingZeros;
    }
    if (name == "countOneBits") {
        return BuiltinFn::kCountOneBits;
    }
    if (name == "countTrailingZeros") {
        return BuiltinFn::kCountTrailingZeros;
    }
    if (name == "cross") {
        return BuiltinFn::kCross;
    }
    if (name == "degrees") {
        return BuiltinFn::kDegrees;
    }
    if (name == "determinant") {
        return BuiltinFn::kDeterminant;
    }
    if (name == "distance") {
        return BuiltinFn::kDistance;
    }
    if (name == "dot") {
        return BuiltinFn::kDot;
    }
    if (name == "dot4I8Packed") {
        return BuiltinFn::kDot4I8Packed;
    }
    if (name == "dot4U8Packed") {
        return BuiltinFn::kDot4U8Packed;
    }
    if (name == "dpdx") {
        return BuiltinFn::kDpdx;
    }
    if (name == "dpdxCoarse") {
        return BuiltinFn::kDpdxCoarse;
    }
    if (name == "dpdxFine") {
        return BuiltinFn::kDpdxFine;
    }
    if (name == "dpdy") {
        return BuiltinFn::kDpdy;
    }
    if (name == "dpdyCoarse") {
        return BuiltinFn::kDpdyCoarse;
    }
    if (name == "dpdyFine") {
        return BuiltinFn::kDpdyFine;
    }
    if (name == "exp") {
        return BuiltinFn::kExp;
    }
    if (name == "exp2") {
        return BuiltinFn::kExp2;
    }
    if (name == "extractBits") {
        return BuiltinFn::kExtractBits;
    }
    if (name == "faceForward") {
        return BuiltinFn::kFaceForward;
    }
    if (name == "firstLeadingBit") {
        return BuiltinFn::kFirstLeadingBit;
    }
    if (name == "firstTrailingBit") {
        return BuiltinFn::kFirstTrailingBit;
    }
    if (name == "floor") {
        return BuiltinFn::kFloor;
    }
    if (name == "fma") {
        return BuiltinFn::kFma;
    }
    if (name == "fract") {
        return BuiltinFn::kFract;
    }
    if (name == "frexp") {
        return BuiltinFn::kFrexp;
    }
    if (name == "fwidth") {
        return BuiltinFn::kFwidth;
    }
    if (name == "fwidthCoarse") {
        return BuiltinFn::kFwidthCoarse;
    }
    if (name == "fwidthFine") {
        return BuiltinFn::kFwidthFine;
    }
    if (name == "insertBits") {
        return BuiltinFn::kInsertBits;
    }
    if (name == "inverseSqrt") {
        return BuiltinFn::kInverseSqrt;
    }
    if (name == "ldexp") {
        return BuiltinFn::kLdexp;
    }
    if (name == "length") {
        return BuiltinFn::kLength;
    }
    if (name == "log") {
        return BuiltinFn::kLog;
    }
    if (name == "log2") {
        return BuiltinFn::kLog2;
    }
    if (name == "max") {
        return BuiltinFn::kMax;
    }
    if (name == "min") {
        return BuiltinFn::kMin;
    }
    if (name == "mix") {
        return BuiltinFn::kMix;
    }
    if (name == "modf") {
        return BuiltinFn::kModf;
    }
    if (name == "normalize") {
        return BuiltinFn::kNormalize;
    }
    if (name == "pack2x16float") {
        return BuiltinFn::kPack2X16Float;
    }
    if (name == "pack2x16snorm") {
        return BuiltinFn::kPack2X16Snorm;
    }
    if (name == "pack2x16unorm") {
        return BuiltinFn::kPack2X16Unorm;
    }
    if (name == "pack4x8snorm") {
        return BuiltinFn::kPack4X8Snorm;
    }
    if (name == "pack4x8unorm") {
        return BuiltinFn::kPack4X8Unorm;
    }
    if (name == "pack4xI8") {
        return BuiltinFn::kPack4XI8;
    }
    if (name == "pack4xU8") {
        return BuiltinFn::kPack4XU8;
    }
    if (name == "pack4xI8Clamp") {
        return BuiltinFn::kPack4XI8Clamp;
    }
    if (name == "pack4xU8Clamp") {
        return BuiltinFn::kPack4XU8Clamp;
    }
    if (name == "pow") {
        return BuiltinFn::kPow;
    }
    if (name == "quantizeToF16") {
        return BuiltinFn::kQuantizeToF16;
    }
    if (name == "radians") {
        return BuiltinFn::kRadians;
    }
    if (name == "reflect") {
        return BuiltinFn::kReflect;
    }
    if (name == "refract") {
        return BuiltinFn::kRefract;
    }
    if (name == "reverseBits") {
        return BuiltinFn::kReverseBits;
    }
    if (name == "round") {
        return BuiltinFn::kRound;
    }
    if (name == "saturate") {
        return BuiltinFn::kSaturate;
    }
    if (name == "select") {
        return BuiltinFn::kSelect;
    }
    if (name == "sign") {
        return BuiltinFn::kSign;
    }
    if (name == "sin") {
        return BuiltinFn::kSin;
    }
    if (name == "sinh") {
        return BuiltinFn::kSinh;
    }
    if (name == "smoothstep") {
        return BuiltinFn::kSmoothstep;
    }
    if (name == "sqrt") {
        return BuiltinFn::kSqrt;
    }
    if (name == "step") {
        return BuiltinFn::kStep;
    }
    if (name == "storageBarrier") {
        return BuiltinFn::kStorageBarrier;
    }
    if (name == "tan") {
        return BuiltinFn::kTan;
    }
    if (name == "tanh") {
        return BuiltinFn::kTanh;
    }
    if (name == "transpose") {
        return BuiltinFn::kTranspose;
    }
    if (name == "trunc") {
        return BuiltinFn::kTrunc;
    }
    if (name == "unpack2x16float") {
        return BuiltinFn::kUnpack2X16Float;
    }
    if (name == "unpack2x16snorm") {
        return BuiltinFn::kUnpack2X16Snorm;
    }
    if (name == "unpack2x16unorm") {
        return BuiltinFn::kUnpack2X16Unorm;
    }
    if (name == "unpack4x8snorm") {
        return BuiltinFn::kUnpack4X8Snorm;
    }
    if (name == "unpack4x8unorm") {
        return BuiltinFn::kUnpack4X8Unorm;
    }
    if (name == "unpack4xI8") {
        return BuiltinFn::kUnpack4XI8;
    }
    if (name == "unpack4xU8") {
        return BuiltinFn::kUnpack4XU8;
    }
    if (name == "workgroupBarrier") {
        return BuiltinFn::kWorkgroupBarrier;
    }
    if (name == "textureBarrier") {
        return BuiltinFn::kTextureBarrier;
    }
    if (name == "textureDimensions") {
        return BuiltinFn::kTextureDimensions;
    }
    if (name == "textureGather") {
        return BuiltinFn::kTextureGather;
    }
    if (name == "textureGatherCompare") {
        return BuiltinFn::kTextureGatherCompare;
    }
    if (name == "textureNumLayers") {
        return BuiltinFn::kTextureNumLayers;
    }
    if (name == "textureNumLevels") {
        return BuiltinFn::kTextureNumLevels;
    }
    if (name == "textureNumSamples") {
        return BuiltinFn::kTextureNumSamples;
    }
    if (name == "textureSample") {
        return BuiltinFn::kTextureSample;
    }
    if (name == "textureSampleBias") {
        return BuiltinFn::kTextureSampleBias;
    }
    if (name == "textureSampleCompare") {
        return BuiltinFn::kTextureSampleCompare;
    }
    if (name == "textureSampleCompareLevel") {
        return BuiltinFn::kTextureSampleCompareLevel;
    }
    if (name == "textureSampleGrad") {
        return BuiltinFn::kTextureSampleGrad;
    }
    if (name == "textureSampleLevel") {
        return BuiltinFn::kTextureSampleLevel;
    }
    if (name == "textureSampleBaseClampToEdge") {
        return BuiltinFn::kTextureSampleBaseClampToEdge;
    }
    if (name == "textureStore") {
        return BuiltinFn::kTextureStore;
    }
    if (name == "textureLoad") {
        return BuiltinFn::kTextureLoad;
    }
    if (name == "inputAttachmentLoad") {
        return BuiltinFn::kInputAttachmentLoad;
    }
    if (name == "atomicLoad") {
        return BuiltinFn::kAtomicLoad;
    }
    if (name == "atomicStore") {
        return BuiltinFn::kAtomicStore;
    }
    if (name == "atomicAdd") {
        return BuiltinFn::kAtomicAdd;
    }
    if (name == "atomicSub") {
        return BuiltinFn::kAtomicSub;
    }
    if (name == "atomicMax") {
        return BuiltinFn::kAtomicMax;
    }
    if (name == "atomicMin") {
        return BuiltinFn::kAtomicMin;
    }
    if (name == "atomicAnd") {
        return BuiltinFn::kAtomicAnd;
    }
    if (name == "atomicOr") {
        return BuiltinFn::kAtomicOr;
    }
    if (name == "atomicXor") {
        return BuiltinFn::kAtomicXor;
    }
    if (name == "atomicExchange") {
        return BuiltinFn::kAtomicExchange;
    }
    if (name == "atomicCompareExchangeWeak") {
        return BuiltinFn::kAtomicCompareExchangeWeak;
    }
    if (name == "subgroupBallot") {
        return BuiltinFn::kSubgroupBallot;
    }
    if (name == "subgroupElect") {
        return BuiltinFn::kSubgroupElect;
    }
    if (name == "subgroupBroadcast") {
        return BuiltinFn::kSubgroupBroadcast;
    }
    if (name == "subgroupBroadcastFirst") {
        return BuiltinFn::kSubgroupBroadcastFirst;
    }
    if (name == "subgroupShuffle") {
        return BuiltinFn::kSubgroupShuffle;
    }
    if (name == "subgroupAdd") {
        return BuiltinFn::kSubgroupAdd;
    }
    if (name == "subgroupExclusiveAdd") {
        return BuiltinFn::kSubgroupExclusiveAdd;
    }
    if (name == "subgroupMul") {
        return BuiltinFn::kSubgroupMul;
    }
    if (name == "subgroupExclusiveMul") {
        return BuiltinFn::kSubgroupExclusiveMul;
    }
    if (name == "subgroupAnd") {
        return BuiltinFn::kSubgroupAnd;
    }
    if (name == "subgroupOr") {
        return BuiltinFn::kSubgroupOr;
    }
    if (name == "subgroupXor") {
        return BuiltinFn::kSubgroupXor;
    }
    if (name == "subgroupMin") {
        return BuiltinFn::kSubgroupMin;
    }
    if (name == "subgroupMax") {
        return BuiltinFn::kSubgroupMax;
    }
    if (name == "subgroupAll") {
        return BuiltinFn::kSubgroupAll;
    }
    if (name == "subgroupAny") {
        return BuiltinFn::kSubgroupAny;
    }
    return BuiltinFn::kNone;
}

const char* str(BuiltinFn i) {
    switch (i) {
        case BuiltinFn::kNone:
            return "<none>";
        case BuiltinFn::kAbs:
            return "abs";
        case BuiltinFn::kAcos:
            return "acos";
        case BuiltinFn::kAcosh:
            return "acosh";
        case BuiltinFn::kAll:
            return "all";
        case BuiltinFn::kAny:
            return "any";
        case BuiltinFn::kArrayLength:
            return "arrayLength";
        case BuiltinFn::kAsin:
            return "asin";
        case BuiltinFn::kAsinh:
            return "asinh";
        case BuiltinFn::kAtan:
            return "atan";
        case BuiltinFn::kAtan2:
            return "atan2";
        case BuiltinFn::kAtanh:
            return "atanh";
        case BuiltinFn::kCeil:
            return "ceil";
        case BuiltinFn::kClamp:
            return "clamp";
        case BuiltinFn::kCos:
            return "cos";
        case BuiltinFn::kCosh:
            return "cosh";
        case BuiltinFn::kCountLeadingZeros:
            return "countLeadingZeros";
        case BuiltinFn::kCountOneBits:
            return "countOneBits";
        case BuiltinFn::kCountTrailingZeros:
            return "countTrailingZeros";
        case BuiltinFn::kCross:
            return "cross";
        case BuiltinFn::kDegrees:
            return "degrees";
        case BuiltinFn::kDeterminant:
            return "determinant";
        case BuiltinFn::kDistance:
            return "distance";
        case BuiltinFn::kDot:
            return "dot";
        case BuiltinFn::kDot4I8Packed:
            return "dot4I8Packed";
        case BuiltinFn::kDot4U8Packed:
            return "dot4U8Packed";
        case BuiltinFn::kDpdx:
            return "dpdx";
        case BuiltinFn::kDpdxCoarse:
            return "dpdxCoarse";
        case BuiltinFn::kDpdxFine:
            return "dpdxFine";
        case BuiltinFn::kDpdy:
            return "dpdy";
        case BuiltinFn::kDpdyCoarse:
            return "dpdyCoarse";
        case BuiltinFn::kDpdyFine:
            return "dpdyFine";
        case BuiltinFn::kExp:
            return "exp";
        case BuiltinFn::kExp2:
            return "exp2";
        case BuiltinFn::kExtractBits:
            return "extractBits";
        case BuiltinFn::kFaceForward:
            return "faceForward";
        case BuiltinFn::kFirstLeadingBit:
            return "firstLeadingBit";
        case BuiltinFn::kFirstTrailingBit:
            return "firstTrailingBit";
        case BuiltinFn::kFloor:
            return "floor";
        case BuiltinFn::kFma:
            return "fma";
        case BuiltinFn::kFract:
            return "fract";
        case BuiltinFn::kFrexp:
            return "frexp";
        case BuiltinFn::kFwidth:
            return "fwidth";
        case BuiltinFn::kFwidthCoarse:
            return "fwidthCoarse";
        case BuiltinFn::kFwidthFine:
            return "fwidthFine";
        case BuiltinFn::kInsertBits:
            return "insertBits";
        case BuiltinFn::kInverseSqrt:
            return "inverseSqrt";
        case BuiltinFn::kLdexp:
            return "ldexp";
        case BuiltinFn::kLength:
            return "length";
        case BuiltinFn::kLog:
            return "log";
        case BuiltinFn::kLog2:
            return "log2";
        case BuiltinFn::kMax:
            return "max";
        case BuiltinFn::kMin:
            return "min";
        case BuiltinFn::kMix:
            return "mix";
        case BuiltinFn::kModf:
            return "modf";
        case BuiltinFn::kNormalize:
            return "normalize";
        case BuiltinFn::kPack2X16Float:
            return "pack2x16float";
        case BuiltinFn::kPack2X16Snorm:
            return "pack2x16snorm";
        case BuiltinFn::kPack2X16Unorm:
            return "pack2x16unorm";
        case BuiltinFn::kPack4X8Snorm:
            return "pack4x8snorm";
        case BuiltinFn::kPack4X8Unorm:
            return "pack4x8unorm";
        case BuiltinFn::kPack4XI8:
            return "pack4xI8";
        case BuiltinFn::kPack4XU8:
            return "pack4xU8";
        case BuiltinFn::kPack4XI8Clamp:
            return "pack4xI8Clamp";
        case BuiltinFn::kPack4XU8Clamp:
            return "pack4xU8Clamp";
        case BuiltinFn::kPow:
            return "pow";
        case BuiltinFn::kQuantizeToF16:
            return "quantizeToF16";
        case BuiltinFn::kRadians:
            return "radians";
        case BuiltinFn::kReflect:
            return "reflect";
        case BuiltinFn::kRefract:
            return "refract";
        case BuiltinFn::kReverseBits:
            return "reverseBits";
        case BuiltinFn::kRound:
            return "round";
        case BuiltinFn::kSaturate:
            return "saturate";
        case BuiltinFn::kSelect:
            return "select";
        case BuiltinFn::kSign:
            return "sign";
        case BuiltinFn::kSin:
            return "sin";
        case BuiltinFn::kSinh:
            return "sinh";
        case BuiltinFn::kSmoothstep:
            return "smoothstep";
        case BuiltinFn::kSqrt:
            return "sqrt";
        case BuiltinFn::kStep:
            return "step";
        case BuiltinFn::kStorageBarrier:
            return "storageBarrier";
        case BuiltinFn::kTan:
            return "tan";
        case BuiltinFn::kTanh:
            return "tanh";
        case BuiltinFn::kTranspose:
            return "transpose";
        case BuiltinFn::kTrunc:
            return "trunc";
        case BuiltinFn::kUnpack2X16Float:
            return "unpack2x16float";
        case BuiltinFn::kUnpack2X16Snorm:
            return "unpack2x16snorm";
        case BuiltinFn::kUnpack2X16Unorm:
            return "unpack2x16unorm";
        case BuiltinFn::kUnpack4X8Snorm:
            return "unpack4x8snorm";
        case BuiltinFn::kUnpack4X8Unorm:
            return "unpack4x8unorm";
        case BuiltinFn::kUnpack4XI8:
            return "unpack4xI8";
        case BuiltinFn::kUnpack4XU8:
            return "unpack4xU8";
        case BuiltinFn::kWorkgroupBarrier:
            return "workgroupBarrier";
        case BuiltinFn::kTextureBarrier:
            return "textureBarrier";
        case BuiltinFn::kTextureDimensions:
            return "textureDimensions";
        case BuiltinFn::kTextureGather:
            return "textureGather";
        case BuiltinFn::kTextureGatherCompare:
            return "textureGatherCompare";
        case BuiltinFn::kTextureNumLayers:
            return "textureNumLayers";
        case BuiltinFn::kTextureNumLevels:
            return "textureNumLevels";
        case BuiltinFn::kTextureNumSamples:
            return "textureNumSamples";
        case BuiltinFn::kTextureSample:
            return "textureSample";
        case BuiltinFn::kTextureSampleBias:
            return "textureSampleBias";
        case BuiltinFn::kTextureSampleCompare:
            return "textureSampleCompare";
        case BuiltinFn::kTextureSampleCompareLevel:
            return "textureSampleCompareLevel";
        case BuiltinFn::kTextureSampleGrad:
            return "textureSampleGrad";
        case BuiltinFn::kTextureSampleLevel:
            return "textureSampleLevel";
        case BuiltinFn::kTextureSampleBaseClampToEdge:
            return "textureSampleBaseClampToEdge";
        case BuiltinFn::kTextureStore:
            return "textureStore";
        case BuiltinFn::kTextureLoad:
            return "textureLoad";
        case BuiltinFn::kInputAttachmentLoad:
            return "inputAttachmentLoad";
        case BuiltinFn::kAtomicLoad:
            return "atomicLoad";
        case BuiltinFn::kAtomicStore:
            return "atomicStore";
        case BuiltinFn::kAtomicAdd:
            return "atomicAdd";
        case BuiltinFn::kAtomicSub:
            return "atomicSub";
        case BuiltinFn::kAtomicMax:
            return "atomicMax";
        case BuiltinFn::kAtomicMin:
            return "atomicMin";
        case BuiltinFn::kAtomicAnd:
            return "atomicAnd";
        case BuiltinFn::kAtomicOr:
            return "atomicOr";
        case BuiltinFn::kAtomicXor:
            return "atomicXor";
        case BuiltinFn::kAtomicExchange:
            return "atomicExchange";
        case BuiltinFn::kAtomicCompareExchangeWeak:
            return "atomicCompareExchangeWeak";
        case BuiltinFn::kSubgroupBallot:
            return "subgroupBallot";
        case BuiltinFn::kSubgroupElect:
            return "subgroupElect";
        case BuiltinFn::kSubgroupBroadcast:
            return "subgroupBroadcast";
        case BuiltinFn::kSubgroupBroadcastFirst:
            return "subgroupBroadcastFirst";
        case BuiltinFn::kSubgroupShuffle:
            return "subgroupShuffle";
        case BuiltinFn::kSubgroupAdd:
            return "subgroupAdd";
        case BuiltinFn::kSubgroupExclusiveAdd:
            return "subgroupExclusiveAdd";
        case BuiltinFn::kSubgroupMul:
            return "subgroupMul";
        case BuiltinFn::kSubgroupExclusiveMul:
            return "subgroupExclusiveMul";
        case BuiltinFn::kSubgroupAnd:
            return "subgroupAnd";
        case BuiltinFn::kSubgroupOr:
            return "subgroupOr";
        case BuiltinFn::kSubgroupXor:
            return "subgroupXor";
        case BuiltinFn::kSubgroupMin:
            return "subgroupMin";
        case BuiltinFn::kSubgroupMax:
            return "subgroupMax";
        case BuiltinFn::kSubgroupAll:
            return "subgroupAll";
        case BuiltinFn::kSubgroupAny:
            return "subgroupAny";
    }
    return "<unknown>";
}

bool IsCoarseDerivative(BuiltinFn f) {
    return f == BuiltinFn::kDpdxCoarse || f == BuiltinFn::kDpdyCoarse ||
           f == BuiltinFn::kFwidthCoarse;
}

bool IsFineDerivative(BuiltinFn f) {
    return f == BuiltinFn::kDpdxFine || f == BuiltinFn::kDpdyFine || f == BuiltinFn::kFwidthFine;
}

bool IsDerivative(BuiltinFn f) {
    return f == BuiltinFn::kDpdx || f == BuiltinFn::kDpdy || f == BuiltinFn::kFwidth ||
           IsCoarseDerivative(f) || IsFineDerivative(f);
}

bool IsTexture(BuiltinFn f) {
    return IsImageQuery(f) ||                                //
           f == BuiltinFn::kTextureGather ||                 //
           f == BuiltinFn::kTextureGatherCompare ||          //
           f == BuiltinFn::kTextureLoad ||                   //
           f == BuiltinFn::kTextureSample ||                 //
           f == BuiltinFn::kTextureSampleBaseClampToEdge ||  //
           f == BuiltinFn::kTextureSampleBias ||             //
           f == BuiltinFn::kTextureSampleCompare ||          //
           f == BuiltinFn::kTextureSampleCompareLevel ||     //
           f == BuiltinFn::kTextureSampleGrad ||             //
           f == BuiltinFn::kTextureSampleLevel ||            //
           f == BuiltinFn::kTextureStore || f == BuiltinFn::kInputAttachmentLoad;
}

bool IsImageQuery(BuiltinFn f) {
    return f == BuiltinFn::kTextureDimensions || f == BuiltinFn::kTextureNumLayers ||
           f == BuiltinFn::kTextureNumLevels || f == BuiltinFn::kTextureNumSamples;
}

bool IsDataPacking(BuiltinFn f) {
    return f == BuiltinFn::kPack4X8Snorm || f == BuiltinFn::kPack4X8Unorm ||
           f == BuiltinFn::kPack2X16Snorm || f == BuiltinFn::kPack2X16Unorm ||
           f == BuiltinFn::kPack2X16Float;
}

bool IsDataUnpacking(BuiltinFn f) {
    return f == BuiltinFn::kUnpack4X8Snorm || f == BuiltinFn::kUnpack4X8Unorm ||
           f == BuiltinFn::kUnpack2X16Snorm || f == BuiltinFn::kUnpack2X16Unorm ||
           f == BuiltinFn::kUnpack2X16Float;
}

bool IsBarrier(BuiltinFn f) {
    return f == BuiltinFn::kWorkgroupBarrier || f == BuiltinFn::kStorageBarrier ||
           f == BuiltinFn::kTextureBarrier;
}

bool IsAtomic(BuiltinFn f) {
    return f == BuiltinFn::kAtomicLoad || f == BuiltinFn::kAtomicStore ||
           f == BuiltinFn::kAtomicAdd || f == BuiltinFn::kAtomicSub || f == BuiltinFn::kAtomicMax ||
           f == BuiltinFn::kAtomicMin || f == BuiltinFn::kAtomicAnd || f == BuiltinFn::kAtomicOr ||
           f == BuiltinFn::kAtomicXor || f == BuiltinFn::kAtomicExchange ||
           f == BuiltinFn::kAtomicCompareExchangeWeak;
}

bool IsPacked4x8IntegerDotProductBuiltin(BuiltinFn f) {
    return f == BuiltinFn::kDot4I8Packed || f == BuiltinFn::kDot4U8Packed ||
           f == BuiltinFn::kPack4XI8 || f == BuiltinFn::kPack4XU8 ||
           f == BuiltinFn::kPack4XI8Clamp || f == BuiltinFn::kPack4XU8Clamp ||
           f == BuiltinFn::kUnpack4XI8 || f == BuiltinFn::kUnpack4XU8;
}

bool HasSideEffects(BuiltinFn f) {
    switch (f) {
        case BuiltinFn::kAtomicAdd:
        case BuiltinFn::kAtomicAnd:
        case BuiltinFn::kAtomicCompareExchangeWeak:
        case BuiltinFn::kAtomicExchange:
        case BuiltinFn::kAtomicMax:
        case BuiltinFn::kAtomicMin:
        case BuiltinFn::kAtomicOr:
        case BuiltinFn::kAtomicStore:
        case BuiltinFn::kAtomicSub:
        case BuiltinFn::kAtomicXor:
        case BuiltinFn::kTextureStore:
            return true;
        default:
            break;
    }
    return false;
}

}  // namespace tint::core
