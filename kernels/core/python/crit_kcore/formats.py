"""crit's kernel core for Triton: number formats (kernels/core/README.md, "formats"). Exact decoders and encoders of the
stored formats, as @triton.jit functions a kernel calls (`from crit_kcore import formats`, then `formats.e2m1(codes)`).

E2M1: a code's FP16 bit pattern (sign at bit 15, magnitude bits at 11..9) is its value times 2^-14, so e2m1_bits16
gives values in that unit (fold 2^14 into a scale); e2m1_pair_cvt is the hardware conversion (compute capability 10
and later, the architecture-specific targets: cvt.rn.f16x2.e2m1x2) and gives the values themselves. Both are exact.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

E2M1_UNIT = 2.0 ** -14   # the unit e2m1_bits16 gives E2M1 values in


@triton.jit
def e2m1_bits16(codes):
    """4-bit codes (the low nibble of an integer tensor; bit 3 the sign) as FP16 values times 2^-14, exactly."""
    bits = ((codes & 7).to(tl.uint16) << 9) | ((codes & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True)


@triton.jit
def e2m1(codes):
    """4-bit codes as FP32 values times 2^-14 (e2m1_bits16 widened), exactly."""
    return e2m1_bits16(codes).to(tl.float32)


@triton.jit
def e2m1_pair_cvt(packed):
    """Bytes of two codes (uint8) as (low nibbles, high nibbles) FP16 values, exactly, by the hardware conversion, four
    bytes an instruction group. Only where cvt.rn.f16x2.e2m1x2 exists."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 r0, r1, r2, r3;
        .reg .b8 b0, b1, b2, b3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 r0, b0;
        cvt.rn.f16x2.e2m1x2 r1, b1;
        cvt.rn.f16x2.e2m1x2 r2, b2;
        cvt.rn.f16x2.e2m1x2 r3, b3;
        prmt.b32 $0, r0, r1, 0x5410;
        prmt.b32 $1, r2, r3, 0x5410;
        prmt.b32 $2, r0, r1, 0x7632;
        prmt.b32 $3, r2, r3, 0x7632;
        }""",
        "=r,=r,=r,=r,r", [packed], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)


@triton.jit
def e4m3_values(x):
    """E4M3 codes (uint8) to their FP32 values, exactly, with integer arithmetic (for GPUs without the conversion);
    the NaN codes read as +-480."""
    b = x.to(tl.int32)
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    unit = ((e + 117) << 23).to(tl.float32, bitcast=True)     # 2^(e - 10)
    mag = tl.where(e == 0, m * 0.001953125, (m + 8.0) * unit)
    return tl.where((b & 128) != 0, -mag, mag)


@triton.jit
def e4m3_codes(x):
    """FP32 values within [-448, 448] to E4M3 codes (uint8), rounded to the nearest code with ties to even, as a
    float8_e4m3fn conversion rounds them."""
    a = tl.abs(x)
    bits = a.to(tl.int32, bitcast=True)
    ex = (bits >> 23) - 127
    # Normal codes: the mantissa rounded to 3 bits (exact scaling by a power of two, then rint: ties to even).
    frac = a * ((127 - ex) << 23).to(tl.float32, bitcast=True)
    mant = libdevice.rint((frac - 1.0) * 8.0).to(tl.int32)
    carry = mant == 8
    ex = tl.where(carry, ex + 1, ex)
    mant = tl.where(carry, 0, mant)
    normal = ((ex + 7) << 3) | mant
    # Below 2^-6 the codes are subnormal: multiples of 2^-9 (code 8 is the smallest normal value, 2^-6).
    sub = libdevice.rint(a * 512.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 128
    return (code | sign).to(tl.uint8)


@triton.jit
def e4m3_to(x, OUT: tl.constexpr, HW_FP8: tl.constexpr):
    """E4M3 codes (uint8) as OUT values, exactly: the hardware conversion where HW_FP8 (compute capability 8.9 on),
    else e4m3_values."""
    if HW_FP8:
        y = x.to(tl.float8e4nv, bitcast=True).to(OUT)
    else:
        y = e4m3_values(x).to(OUT)
    return y


@triton.jit
def e8m0_values(e):
    """E8M0 scale bytes as FP32 scales 2^(e - 127): byte 0 is 2^-127 (an FP32 subnormal), byte 255 NaN."""
    b = e.to(tl.int32)
    bits = tl.where(b == 0, 0x00400000, tl.where(b == 255, 0x7FC00000, b << 23))
    return bits.to(tl.float32, bitcast=True)


# ------------------------------------------------------------------------------------------- weight formats
#
# The weight formats crit's projection kernels read, by the codes kernels key them with (decode-gemv's gemv-cuda and
# gemv-triton, mma-rows): formats.cuh's decoders, value for value. NVFP4: E2M1 codes two a byte (element 2i in the low
# nibble of byte i) times an E4M3 scale per 16, a row factor after the sum. FP8: E4M3 codes, an FP32 scale a row after
# the sum. MXFP8: E4M3 codes times an E8M0 power of two per 32. BF16. INT4 with a zero point a group (AWQ, GPTQ in
# crit's rows): unsigned codes two a byte, (code - zero) * an FP16 scale. Every value below is exact in FP32, and so is
# its product with a 16-bit activation.
NVFP4, FP8, MXFP8, BF16, INT4_ZP = 0, 1, 2, 3, 4


@triton.jit
def bf16_values(bits):
    """BF16 bit patterns (uint16 or int16) as their FP32 values, exactly."""
    return (bits.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def f16_values(bits):
    """FP16 bit patterns (uint16 or int16) as their FP32 values, exactly."""
    return bits.to(tl.uint16, bitcast=True).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def int4_zp_offsets(codes, zero):
    """Unsigned 4-bit codes (the low nibble) minus their zero point (an integer 0 to 16) as FP32 integers, exactly:
    the part of INT4 with a zero point a scale multiplies (formats.cuh's int4_zp_value is this times the scale)."""
    return ((codes & 15).to(tl.int32) - zero.to(tl.int32)).to(tl.float32)


@triton.jit
def int4_zp_values(codes, zero, scale):
    """(code - zero) * scale in FP32, exactly (|code - zero| <= 16: 4 significant bits; an FP16 scale: 11):
    formats.cuh's int4_zp_value."""
    return int4_zp_offsets(codes, zero) * scale


@triton.jit
def split_bf16(v):
    """FP32 values with at most 16 significant bits as BF16 (hi, lo), hi + lo == v exactly: hi v rounded to BF16, lo
    the rest (formats.cuh's int4x8_zp_to_bf16x2_split). Two BF16 dots of the same activations, one with hi and one with
    lo, multiply v exactly; a value that fits BF16 has lo 0."""
    hi = v.to(tl.bfloat16)
    lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo
