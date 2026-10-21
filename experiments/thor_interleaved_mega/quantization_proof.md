# Exact integer quantization using a guarded reciprocal

Only the final `value / scale` used for INT8 quantization changes. The scale
calculation, SiLU division, projection rounding and output layouts are unchanged.

Let RN mean IEEE binary32 round-to-nearest, ties-to-even, and `u=2^-24`.
The kernel computes `A=max(amax(abs(values)),float32(1e-12))`, then
`s=RN(A/127)`. For finite A, both s and `r=RN(1/s)` are positive normal floats:

- `7.874015736577134e-15 <= s <= 2.6793887059314005e36`.
- `3.732194563860763e-37 <= r <= 1.2700000190464e14`.
- Every group value satisfies `abs(v)<=A` and thus
  `abs(v/s)<=127/(1-u)<128`.

Set `a=RN(v*r)`, `d=RN(v/s)`. The reciprocal, multiplication and division each
introduce at most one binary32 rounding error. Allowing an absolute `2^-150`
term for subnormal multiplication/division gives the conservative bound

```
abs(a-d) < 128*(3*u+u*u) + 2^-149
         = 2.288818404849735e-5.
```

The exact real product before RN is bounded far below overflow. If a quotient
or product is subnormal, it is far below 0.5, so both integer roundings are zero.

Nearest-integer rounding can differ only if a half-integer boundary lies between
a and d, including when d is exactly a tie. Therefore a must lie within the bound
above of a half-integer. Let `q=nearbyintf(a)` and compute
`distance=abs(abs(a-q)-0.5)`. Near a half-integer the float subtractions are exact
by Sterbenz's lemma: a and q have the same sign/magnitude within a factor of two,
and `abs(a-q)` is near 0.5. Away from a half-integer, rounding of this diagnostic
cannot move a distance above 0.25 into the tiny guard region.

The implementation uses a strictly wider binary32 guard of 1e-4 and falls back
to the original `nearbyintf(__fdiv_rn(v,s))` when `distance<=1e-4`. Thus every
possible different integer rounding, including either direction of a tie to even,
uses the original division. Else `nearbyintf(a)` is the same integer. The original
clamp to [-127,127] and int conversion remain identical.

Nonfinite cases retain the original behavior:

- Finite v and infinite s: reciprocal and quotient are signed zero; both yield
  integer zero.
- Infinite v and infinite s, or NaN v: a is nonfinite, triggering the original
  division. CUDA fminf/fmaxf clipping then receives the original rounded value.
- A cannot become zero or NaN: the positive 1e-12 initializer and fmaxf reduction
  establish the lower bound even when individual activation values are NaN.

A standalone CUDA helper test compares both the original and candidate rounded
floats / clipped integers on every value in reproducible CPU-generated vectors.
It additionally compares original CUDA clipped integers to an independent CPU
binary32 reference. The CUDA helper audit also passed all 8,326,770 cases with zero mismatches.
Complete FFN checks remain necessary for each compiled kernel variant.

The bounded CPU audit covers 8,326,770 values: all half-integer boundaries from
0.5 through126.5, both signs and +/-3 adjacent representable input values over
4,355 scales spanning exponents/extrema; ordinary interior values; +/-amax;
zeros/subnormal inputs; and nonfinite cases. It reports zero mismatches and
maximum observed `abs(a-d)=7.62939453125e-6`. Most test values deliberately fall
inside the fallback region; the independent uniform-ratio subset falls back on
105/557,440 values (0.0188%). CPU results supplement the proof and do not replace
GPU correctness or performance measurements.
