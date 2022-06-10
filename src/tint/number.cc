// Copyright 2022 The Tint Authors.
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

#include "src/tint/number.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ostream>

#include "src/tint/debug.h"

namespace tint {

std::ostream& operator<<(std::ostream& out, ConversionFailure failure) {
    switch (failure) {
        case ConversionFailure::kExceedsPositiveLimit:
            return out << "value exceeds positive limit for type";
        case ConversionFailure::kExceedsNegativeLimit:
            return out << "value exceeds negative limit for type";
    }
    return out << "<unknown>";
}

f16::type f16::Quantize(f16::type value) {
    if (value > kHighest) {
        return std::numeric_limits<f16::type>::infinity();
    }
    if (value < kLowest) {
        return -std::numeric_limits<f16::type>::infinity();
    }
    // Below value must be within the finite range of a f16.
    // Assert we use binary32 (i.e. float) as underlying type, which has 4 bytes.
    static_assert(std::is_same<f16::type, float>());
    const uint32_t sign_mask = 0x80000000u;      // Mask for the sign bit
    const uint32_t exponent_mask = 0x7f800000u;  // Mask for 8 exponent bits

    uint32_t u32;
    memcpy(&u32, &value, 4);

    if ((u32 & ~sign_mask) == 0) {
        return value;                // +/- zero
    }
    if ((u32 & exponent_mask) == exponent_mask) {  // exponent all 1's
        return value;                        // inf or nan
    }

    // We are now going to quantize a f32 number into subnormal f16 and store the result value back
    // into a f32 variable. Notice that all subnormal f16 values are just normal f32 values. Below
    // will show that we can do this quantization by just masking out 13 or more lowest mantissa
    // bits of the original f32 number.
    //
    // Note:
    // f32 has 1 sign bit, 8 exponent bits for biased exponent (i.e. unbiased exponent + 127), and
    // 23 mantissa bits. Binary form: s_eeeeeeee_mmmmmmmmmmmmmmmmmmmmmmm
    // f16 has 1 sign bit, 5 exponent bits for biased exponent (i.e. unbiased exponent + 15), and
    // 10 mantissa bits. Binary form: s_eeeee_mmmmmmmmmm
    // The largest finite f16 number has a biased exponent of 11110 in binary, or 30 decimal, and so
    // a unbiased exponent of 30 - 15 = 15.
    // The smallest finite f16 number has a biased exponent of 00001 in binary, or 1 decimal, and so
    // a unbiased exponent of 1 - 15 = -14.
    //
    // We may follow the argument below:
    // 1. All normal or subnormal f16 values, range from 0x1.p-24 to 0x1.ffcp15, are exactly
    // representable by normal f32 number.
    //   1.1. We can denote the set of all f32 number that are exact representation of finite f16
    //   values by `R`.
    //   1.2. We can do the quantization by mapping a normal f32 value v (in the f16 finite range)
    //   to a certain f32 number v' in the set R, which is the largest (by the meaning of absolute
    //   value) one among all values in R that are no larger than v.
    // 2. We can decide whether a given normal f32 number v is in the set R, by looking at its
    // mantissa bits and biased exponent `e`. Recall that biased exponent e is unbiased exponent +
    // 127, and in the range of 1 to 254 for normal f32 number.
    //   2.1. If e >= 143, i.e. abs(v) >= 2^16 > f16::kHighest = 0x1.ffcp15, v is larger than any
    //   finite f16 value and can not be in set R.
    //   2.2. If 142 >= e >= 113, or f16::kHighest >= abs(v) >= f16::kSmallest = 2^-14, v falls in
    //   the range of normal f16 values. In this case, v is in the set R iff the lowest 13 mantissa
    //   bits are all 0. (See below for proof)
    //     2.2.1. If we let v' be v with lowest 13 mantissa bits masked to 0, v' will be in set R
    //     and the largest one in set R that no larger than v. Such v' is the quantized value of v.
    //   2.3. If 112 >= e >= 103, i.e. 2^-14 > abs(v) >= f16::kSmallestSubnormal = 2^-24, v falls in
    //   the range of subnormal f16 values. In this case, v is in the set R iff the lowest 126-e
    //   mantissa bits are all 0. Notice that 126-e is in range 14 to 23, inclusive. (See below for
    //   proof)
    //     2.3.1. If we let v' be v with lowest 126-e mantissa bits masked to 0, v' will be in set R
    //     and the largest on in set R that no larger than v. Such v' is the quantized value of v.
    //   2.4. If 2^-24 > abs(v) > 0, i.e. 103 > e, v is smaller than any finite f16 value and not
    //   equal to 0.0, thus can not be in set R.
    //   2.5. If abs(v) = 0, v is in set R and is just +-0.0.
    //
    // Proof for 2.2:
    // Any normal f16 number, in binary form, s_eeeee_mmmmmmmmmm, has value
    //      (s==0?1:-1)*(1+uint(mmmmm_mmmmm)*(2^-10))*2^(uint(eeeee)-15)
    // in which unit(bbbbb) means interprete binary pattern "bbbbb" as unsigned binary number,
    // and we have 1 <= uint(eeeee) <= 30.
    // This value is equal to a normal f32 number with binary
    //      s_EEEEEEEE_mmmmmmmmmm0000000000000
    // where uint(EEEEEEEE) = uint(eeeee) + 112, so that unbiased exponent keep unchanged
    //      uint(EEEEEEEE) - 127 = uint(eeeee) - 15
    // and its value is
    //         (s==0?1:-1)*(1+uint(mmmmm_mmmmm_00000_00000_000)*(2^-23))*2^(uint(EEEEEEEE)-127)
    //      == (s==0?1:-1)*(1+uint(mmmmm_mmmmm)*(2^-10))*2^(uint(eeeee)-15)
    // Notice that uint(EEEEEEEE) is in range [113, 142], showing that it is a normal f32 number.
    // So we proof that any normal f16 number can be exactly representd by a normal f32 number
    // with biased exponent in range [113,142] and the lowest 13 mantissa bits 0.
    // On the other hand, since mantissa bits mmmmmmmmmm are arbitrary, the value of any f32
    // that has a biased exponent in range [113, 142] and lowest 13 mantissa bits zero is equal
    // to a normal f16 value. Hence we proof 2.2.
    //
    // Proof for 2.3:
    // Any subnormal f16 number has a binary form of s_00000_mmmmmmmmmm, and its value is
    // (s==0?1:-1)*uint(mmmmmmmmmm)*(2^-10)*(2^-14) = (s==0?1:-1)*uint(mmmmmmmmmm)*(2^-24).
    // We discuss on bit pattern of mantissa bits mmmmmmmmmm.
    //   Case 1: mantissa bits has no leading zero bit, s_00000_1mmmmmmmmm
    //      In this case the value is
    //              (s==0?1:-1)*uint(1mmmm_mmmmm)*(2^-10)*(2^-14)
    //          ==  (s==0?1:-1)*(uint(1_mmmmm_mmmm)*(2^-9))*(2^-15)
    //          ==  (s==0?1:-1)*(1+uint(mmmmm_mmmm)*(2^-9))*(2^-15)
    //          ==  (s==0?1:-1)*(1+uint(mmmmm_mmmm0_00000_00000_000)*(2^-23))*(2^-15)
    //      which is equal to the value of normal f32 number
    //          s_EEEEEEEE_mmmmm_mmmm0_00000_00000_000
    //      where uint(EEEEEEEE) = -15 + 127 = 112. Hence we proof that any subnormal f16 number
    //      with no leading zero mantissa bit can be exactly represented by a f32 number with
    //      biased exponent 112 and the lowest 14 mantissa bits zero, and the value of any f32
    //      number with biased exponent 112 and the lowest 14 mantissa bits zero are equal to a
    //      subnormal f16 number with no leading zero mantissa bit.
    //   Case 2: mantissa bits has 1 leading zero bit, s_00000_01mmmmmmmm
    //      In this case the value is
    //              (s==0?1:-1)*uint(01mmm_mmmmm)*(2^-10)*(2^-14)
    //          ==  (s==0?1:-1)*(uint(01_mmmmm_mmm)*(2^-8))*(2^-16)
    //          ==  (s==0?1:-1)*(1+uint(mmmmm_mmm)*(2^-8))*(2^-16)
    //          ==  (s==0?1:-1)*(1+uint(mmmmm_mmm00_00000_00000_000)*(2^-23))*(2^-16)
    //      which is equal to the value of normal f32 number
    //          s_EEEEEEEE_mmmmm_mmm00_00000_00000_000
    //      where uint(EEEEEEEE) = -16 + 127 = 111. Hence we proof that any subnormal f16 number
    //      with 1 leading zero mantissa bit can be exactly represented by a f32 number with
    //      biased exponent 111 and the lowest 15 mantissa bits zero, and the value of any f32
    //      number with biased exponent 111 and the lowest 15 mantissa bits zero are equal to a
    //      subnormal f16 number with 1 leading zero mantissa bit.
    //   Case 3 to case 8: ......
    //   Case 9: mantissa bits has 8 leading zero bit, s_00000_000000001m
    //      In this case the value is
    //              (s==0?1:-1)*uint(00000_0001m)*(2^-10)*(2^-14)
    //          ==  (s==0?1:-1)*(uint(000000001_m)*(2^-1))*(2^-23)
    //          ==  (s==0?1:-1)*(1+uint(m)*(2^-1))*(2^-23)
    //          ==  (s==0?1:-1)*(1+uint(m0000_00000_00000_00000_000)*(2^-23))*(2^-23)
    //      which is equal to the value of normal f32 number
    //          s_EEEEEEEE_m0000_00000_00000_00000_000
    //      where uint(EEEEEEEE) = -23 + 127 = 104. Hence we proof that any subnormal f16 number
    //      with 8 leading zero mantissa bit can be exactly represented by a f32 number with
    //      biased exponent 104 and the lowest 22 mantissa bits zero, and the value of any f32
    //      number with biased exponent 104 and the lowest 22 mantissa bits zero are equal to a
    //      subnormal f16 number with 8 leading zero mantissa bit.
    //   Case 10: mantissa bits has 9 leading zero bit, s_00000_0000000001
    //      In this case the value is just +-2^-24 = +-0x1.0p-24,
    //      the f32 number has biased exponent 103 and all 23 mantissa bits zero.
    //   Case 11: mantissa bits has 10 leading zero bit, s_00000_0000000000, just 0.0
    // Concluding all these case, we proof that any subnormal f16 number with N leading zero
    // mantissa bit can be exactly represented by a f32 number with biased exponent 112-N and the
    // lowest 14+N mantissa bits zero, and the value of any f32 number with biased exponent 112-N (=
    // e) and the lowest 14+N (= 126-e) mantissa bits zero are equal to a subnormal f16 number with
    // N leading zero mantissa bit. N is in range [0, 9], so the f32 number's biased exponent e is
    // in range [103, 112], or unbiased exponent in [-24, -15].

    float abs_value = std::fabs(value);
    if (abs_value >= kSmallest) {
        // Value falls in the normal f16 range, quantize it to a normal f16 value by masking out
        // lowest 13 mantissa bits.
        u32 = u32 & ~((1u << 13) - 1);
    } else if (abs_value >= kSmallestSubnormal) {
        // Value should be quantized to a subnormal f16 value.

        // Get the biased exponent `e` of f32 value, e.g. value 127 representing exponent 2^0.
        uint32_t biased_exponent_original = (u32 & exponent_mask) >> 23;
        // Since we ensure that kSmallest = 0x1f-14 > abs(value) >= kSmallestSubnormal = 0x1f-24,
        // value will have a unbiased exponent in range -24 to -15 (inclusive), and the
        // corresponding biased exponent in f32 is in range 103 to 112 (inclusive).
        TINT_ASSERT(Semantic,
                    (103 <= biased_exponent_original) && (biased_exponent_original <= 112));

        // As we have proved, masking out the lowest 126-e mantissa bits of input value will result
        // in a valid subnormal f16 value, which is exactly the required quantization result.
        uint32_t discard_bits = 126 - biased_exponent_original;  // In range 14 to 23 (inclusive)
        TINT_ASSERT(Semantic, (14 <= discard_bits) && (discard_bits <= 23));
        uint32_t discard_mask = (1u << discard_bits) - 1;
        u32 = u32 & ~discard_mask;
    } else {
        // value is too small that it can't even be represented as subnormal f16 number. Quantize
        // to zero.
        return value > 0 ? 0.0 : -0.0;
    }
    memcpy(&value, &u32, 4);
    return value;
}

}  // namespace tint
