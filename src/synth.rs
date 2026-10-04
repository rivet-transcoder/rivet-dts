//! Reconstruction filters of the DTS core: the 32-band QMF synthesis bank
//! (ETSI TS 102 114 Annex C.3.6) and the LFE interpolation FIR (C.3.7).
//!
//! Both are written to follow the annex pseudocode line by line, with the
//! per-channel state (`raX`, `raZ`, the LFE history) held in the structs so
//! it persists across subframes and frames as the pseudocode assumes.

use std::sync::LazyLock;

use super::tables;

/// Number of subbands in the core filter bank.
pub const NUM_SUBBANDS: usize = 32;

/// The `rScale` of `QMFInterpolation`, which C.3.6 applies to every output
/// sample but never assigns. The structure as printed — post-scaling terms of
/// `0.25 / (2 cos)` and a prototype whose 512 taps sum to 1 — passes a
/// constant on subband 0 through at 1/(128·√2) of its value, while the
/// subband samples are already on the PCM scale (the D.1 scale factors top
/// out at 2^23, the LFE takes the same tables with no such factor). So the
/// reconstruction is unity-gain only with this factor. The round trips
/// (`tests/encoder_roundtrip.rs`) hold it to ±0.005 dB, and an independent
/// decoder run as a black box agrees on the level (`tests/dcadec.rs`).
const RECONSTRUCTION_GAIN: f64 = 128.0 * std::f64::consts::SQRT_2;

/// `PreCalCosMod()` of Annex C.3.6: 16×16 + 16×16 cosine terms followed by
/// the 16 + 16 post-scaling terms, in the order the interpolation consumes
/// them (`raCosMod[j++]`).
static COS_MOD: LazyLock<[f64; 544]> = LazyLock::new(|| {
    use std::f64::consts::PI;
    let mut m = [0.0f64; 544];
    let mut j = 0;
    for k in 0..16 {
        for i in 0..16 {
            m[j] = (((2 * i + 1) * (2 * k + 1)) as f64 * PI / 64.0).cos();
            j += 1;
        }
    }
    for k in 0..16 {
        for i in 0..16 {
            m[j] = ((i * (2 * k + 1)) as f64 * PI / 32.0).cos();
            j += 1;
        }
    }
    for k in 0..16 {
        m[j] = 0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).cos());
        j += 1;
    }
    for k in 0..16 {
        m[j] = -0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).sin());
        j += 1;
    }
    m
});

/// [`COS_MOD`]'s two 16×16 matrices stored by input (`[i][k]` holds
/// `raCosMod[16k + i]`, then `raCosMod[256 + 16k + i]`), so that the sums
/// for all `k` can run together.
static COS_MOD_T: LazyLock<[[[f64; 16]; 16]; 2]> = LazyLock::new(|| {
    std::array::from_fn(|m| {
        std::array::from_fn(|i| std::array::from_fn(|k| COS_MOD[256 * m + 16 * k + i]))
    })
});

crate::simd::avx2_or_portable! {
    /// The arithmetic of `QMFInterpolation` (Annex C.3.6) for one block:
    /// modulation into the newest 32 entries of `x`, then the prototype into
    /// `z`. Every sum is the pseudocode's, term by term in its order, but
    /// the loops run across the outputs (16 modulation sums, 32 filter
    /// outputs) so that they vectorise.
    fn qmf_interpolate(x: &mut [f64; 512], z: &mut [f64; 64], xin: &[f64; NUM_SUBBANDS], coeff: &[f32; 512]) {
        let cm = &*COS_MOD;
        let ct = &*COS_MOD_T;
        // Cosine modulation → SUM / DIFF.
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        for i in 0..16 {
            let s = xin[2 * i] + xin[2 * i + 1];
            for k in 0..16 {
                a[k] += s * ct[0][i][k];
            }
        }
        for i in 0..16 {
            let v = if i > 0 { xin[2 * i] + xin[2 * i - 1] } else { xin[0] };
            for k in 0..16 {
                b[k] += v * ct[1][i][k];
            }
        }
        // Store history: the new 32 entries of raX.
        for k in 0..16 {
            x[k] = cm[512 + k] * (a[k] + b[k]);
        }
        for k in 0..16 {
            x[32 - k - 1] = cm[528 + k] * (a[k] - b[k]);
        }
        // Multiply by the prototype filter (8 taps of 64 per output).
        let mut acc = [0.0f64; 32];
        let mut acc2 = [0.0f64; 32];
        for jj in (0..512).step_by(64) {
            for i in 0..32 {
                let k = 31 - i;
                acc[i] += coeff[i + jj] as f64 * (x[i + jj] - x[jj + k]);
                acc2[i] += coeff[32 + i + jj] as f64 * (-x[i + jj] - x[jj + k]);
            }
        }
        for i in 0..32 {
            z[i] += acc[i];
            z[32 + i] += acc2[i];
        }
    }
}

/// One primary channel's QMF synthesis state.
pub struct Qmf {
    /// `raX`: the 512-sample modulated history, newest 32 first.
    x: [f64; 512],
    /// `raZ`: the 64-sample overlap accumulator; the first 32 are output,
    /// the second 32 carry into the next block.
    z: [f64; 64],
}

impl Default for Qmf {
    fn default() -> Self {
        Self::new()
    }
}

impl Qmf {
    pub fn new() -> Self {
        Self {
            x: [0.0; 512],
            z: [0.0; 64],
        }
    }

    /// `QMFInterpolation` for one subband sample vector: 32 subband samples
    /// in, 32 PCM samples out on the same scale (see [`RECONSTRUCTION_GAIN`]).
    /// `perfect` selects the FILTS == 1 prototype.
    pub fn synthesize(&mut self, xin: &[f64; NUM_SUBBANDS], perfect: bool, out: &mut [f64; 32]) {
        let coeff: &[f32; 512] = if perfect {
            &tables::QMF_FIR_PERFECT
        } else {
            &tables::QMF_FIR_NON_PERFECT
        };
        qmf_interpolate(&mut self.x, &mut self.z, xin, coeff);
        for (o, z) in out.iter_mut().zip(&self.z[..32]) {
            *o = z * RECONSTRUCTION_GAIN;
        }

        // Update working arrays.
        self.x.copy_within(0..480, 32);
        self.z.copy_within(32..64, 0);
        self.z[32..].fill(0.0);
    }
}

/// The LFE channel's interpolation state: the last `512 / factor - 1`
/// decimated samples, newest last.
pub struct LfeInterp {
    hist: [f64; 8],
}

impl Default for LfeInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl LfeInterp {
    pub fn new() -> Self {
        Self { hist: [0.0; 8] }
    }

    /// `InterpolationFIR`: each decimated sample yields `factor` (64 or 128)
    /// PCM samples, appended to `out`.
    pub fn interpolate(&mut self, decimated: &[f64], factor: usize, out: &mut Vec<f64>) {
        debug_assert!(factor == 64 || factor == 128);
        let coeff: &[f32; 512] = if factor == 128 {
            &tables::LFE_FIR_128X
        } else {
            &tables::LFE_FIR_64X
        };
        let taps = 512 / factor; // 4 or 8 decimated samples per output
        for &s in decimated {
            // hist[7] is the newest sample (rLFE[n]), hist[6] is rLFE[n-1], …
            self.hist.copy_within(1..8, 0);
            self.hist[7] = s;
            for k in 0..factor {
                let mut acc = 0.0f64;
                for j in 0..taps {
                    acc += self.hist[7 - j] * coeff[k + j * factor] as f64;
                }
                out.push(acc);
            }
        }
    }
}

/// Subbands of the X96 synthesis bank.
pub const NUM_SUBBANDS_64: usize = 64;

/// The X96 64-band synthesis bank (6.2.3, prototype D.9).
///
/// The specification prints the 1 024-tap prototype and says the bank is a
/// cosine-modulated one obtained by modulating it, but gives no structure
/// for it (C.3.6 covers only the 32-band one). It is written here in direct
/// form from the 32-band bank's own direct form: C.3.6 with its `rScale` is
/// exactly `y[n] = Σ_k x_k · 64 · s_k · g[n] · cos(π/32 · (k+½) · (n+16.5))`
/// with `s_k` = sign(cos((2k+1)π/4)) and `g` the D.8 prototype with every
/// second block of 64 taps negated back (verified tap for tap by
/// `c36_is_the_cosine_modulated_bank` below). The 64-band bank is the same
/// formula with 64 for 32 (`cos(π/64 · (k+½) · (n+32.5))`, gain 128, D.9
/// with every second block of 128 taps negated back, as 6.2.4.7 describes)
/// — the generalisation under which, as 6.2.3 requires, core subband
/// samples placed in the lower 32 bands synthesise to the core's PCM
/// interpolated to the doubled rate (`x96_bank_interpolates_the_core_bank`).
pub struct Qmf64 {
    /// The last 16 modulated vectors, a ring with the newest at `head`:
    /// the `i`-th newest is `v[(head + i) % 16]`, `v[..][j]` for j < 128,
    /// with `v[j + 128] = −v[j]`.
    v: Vec<[f64; 128]>,
    head: usize,
}

/// [`COS_MOD_64`] stored by `k`: `[k][j]`.
static COS_MOD_64_T: LazyLock<Vec<[f64; 128]>> = LazyLock::new(|| {
    (0..64)
        .map(|k| std::array::from_fn(|j| COS_MOD_64[j][k]))
        .collect()
});

crate::simd::avx2_or_portable! {
    /// The X96 synthesis' modulation, `v[j] = Σ_k cm[j][k] x[k]` in order
    /// of `k` (from -0.0, as `Iterator::sum` starts), for all `j` at once.
    fn modulate64(xin: &[f64; NUM_SUBBANDS_64], v: &mut [f64; 128]) {
        let ct = &*COS_MOD_64_T;
        *v = [-0.0; 128];
        for (ck, &xk) in ct.iter().zip(xin) {
            for j in 0..128 {
                v[j] += ck[j] * xk;
            }
        }
    }
}

crate::simd::avx2_or_portable! {
    /// The X96 synthesis' window: `out[t] = Σ_i g[t + 64i] · ±v_i[..]` over
    /// the 16 newest vectors in order, for all `t` at once.
    fn window64(v: &[[f64; 128]], head: usize, out: &mut [f64; 64]) {
        let g = &*PROTO_64;
        let mut acc = [0.0f64; 64];
        for i in 0..16 {
            let vi = &v[(head + i) % 16];
            // n = t + 64 i: one half of v_i, negated in odd 128-blocks.
            let (base, neg) = (64 * (i % 2), (i / 2) % 2 == 1);
            let gi = &g[64 * i..64 * i + 64];
            let vs = &vi[base..base + 64];
            if neg {
                for t in 0..64 {
                    acc[t] += gi[t] * -vs[t];
                }
            } else {
                for t in 0..64 {
                    acc[t] += gi[t] * vs[t];
                }
            }
        }
        *out = acc;
    }
}

/// `s_k · cos(π/64 · (k+½) · (j+32.5))` for j < 128, k < 64.
static COS_MOD_64: LazyLock<Vec<[f64; 64]>> = LazyLock::new(|| {
    use std::f64::consts::PI;
    (0..128)
        .map(|j| {
            std::array::from_fn(|k| {
                let s = if ((2 * k + 1) as f64 * PI / 4.0).cos() > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                s * (PI / 64.0 * (k as f64 + 0.5) * (j as f64 + 32.5)).cos()
            })
        })
        .collect()
});

/// D.9 with the printed sign changes undone, × 128.
static PROTO_64: LazyLock<[f64; 1024]> = LazyLock::new(|| {
    std::array::from_fn(|n| {
        let g = tables::X96_QMF_FIR[n];
        128.0 * if (n / 128) % 2 == 1 { -g } else { g }
    })
});

impl Default for Qmf64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Qmf64 {
    pub fn new() -> Self {
        Self {
            v: vec![[0.0; 128]; 16],
            head: 0,
        }
    }

    /// 64 subband samples in, 64 PCM samples (at twice the core rate) out.
    pub fn synthesize(&mut self, xin: &[f64; NUM_SUBBANDS_64], out: &mut [f64; 64]) {
        // The newest vector goes in front of the others (a ring).
        self.head = (self.head + 15) % 16;
        modulate64(xin, &mut self.v[self.head]);
        window64(&self.v, self.head, out);
    }
}

/// Table 6-11: the X96 LFE 2× interpolation filter, scaled by 2.
#[allow(clippy::excessive_precision)] // as printed
const LFE_2X: [f64; 5] = [
    1.2553677676342990e-1,
    4.9999913800216800e-1,
    7.4892817046880420e-1,
    4.9999913800216800e-1,
    1.2553677676342990e-1,
];

/// The X96 LFE 2× interpolator (6.2.4.7): zero-stuff, then Table 6-11.
#[derive(Default)]
pub struct Lfe2x {
    /// The last two input samples, newest first.
    hist: [f64; 2],
}

impl Lfe2x {
    /// Each input sample yields two output samples, appended to `out`.
    pub fn interpolate(&mut self, input: &[f64], out: &mut Vec<f64>) {
        for &x in input {
            // Upsampled u = [x, 0, h0, 0, h1, …]: even output taps hit
            // samples, odd ones the zeros.
            let even = LFE_2X[0] * x + LFE_2X[2] * self.hist[0] + LFE_2X[4] * self.hist[1];
            let odd = LFE_2X[1] * x + LFE_2X[3] * self.hist[0];
            out.push(even);
            out.push(odd);
            self.hist = [x, self.hist[0]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * scale
            })
            .collect()
    }

    /// `QMFInterpolation`'s arithmetic as the pseudocode orders it, one
    /// output at a time.
    fn literal_interpolate(
        x: &mut [f64; 512],
        z: &mut [f64; 64],
        xin: &[f64; NUM_SUBBANDS],
        coeff: &[f32; 512],
    ) {
        let cm = &*COS_MOD;
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        let mut j = 0;
        for ak in a.iter_mut() {
            for i in 0..16 {
                *ak += (xin[2 * i] + xin[2 * i + 1]) * cm[j];
                j += 1;
            }
        }
        for bk in b.iter_mut() {
            for i in 0..16 {
                let v = if i > 0 {
                    xin[2 * i] + xin[2 * i - 1]
                } else {
                    xin[0]
                };
                *bk += v * cm[j];
                j += 1;
            }
        }
        for k in 0..16 {
            x[k] = cm[j] * (a[k] + b[k]);
            j += 1;
        }
        for k in 0..16 {
            x[32 - k - 1] = cm[j] * (a[k] - b[k]);
            j += 1;
        }
        for i in 0..32 {
            let k = 31 - i;
            let mut acc = 0.0f64;
            let mut acc2 = 0.0f64;
            for jj in (0..512).step_by(64) {
                acc += coeff[i + jj] as f64 * (x[i + jj] - x[jj + k]);
                acc2 += coeff[32 + i + jj] as f64 * (-x[i + jj] - x[jj + k]);
            }
            z[i] += acc;
            z[32 + i] += acc2;
        }
    }

    /// The vectorised synthesis kernels equal their literal forms bit for
    /// bit, on random state and input over a wide range of scales.
    #[test]
    fn synthesis_kernels_match_their_literal_form() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (seed, scale) in [
            (1u64, 1.0),
            (2, 8_388_608.0),
            (3, 1e-300),
            (4, 0.0),
            (5, 1e300),
        ] {
            for coeff in [&tables::QMF_FIR_PERFECT, &tables::QMF_FIR_NON_PERFECT] {
                let xin: [f64; NUM_SUBBANDS] = noise(32, seed, scale).try_into().unwrap();
                let mut x1: [f64; 512] = noise(512, seed + 9, scale).try_into().unwrap();
                let mut z1: [f64; 64] = noise(64, seed + 99, scale).try_into().unwrap();
                let (mut x2, mut z2) = (x1, z1);
                qmf_interpolate(&mut x1, &mut z1, &xin, coeff);
                literal_interpolate(&mut x2, &mut z2, &xin, coeff);
                assert_eq!(
                    (bits(&x1), bits(&z1)),
                    (bits(&x2), bits(&z2)),
                    "seed {seed}"
                );
            }
            // X96: modulation, and the window over a ring at every rotation.
            let xin: [f64; NUM_SUBBANDS_64] = noise(64, seed + 7, scale).try_into().unwrap();
            let mut v = [0.0f64; 128];
            modulate64(&xin, &mut v);
            let want: Vec<f64> = (0..128)
                .map(|j| COS_MOD_64[j].iter().zip(&xin).map(|(c, x)| c * x).sum())
                .collect();
            assert_eq!(bits(&v), bits(&want), "modulate64 seed {seed}");
            let ring: Vec<[f64; 128]> = (0..16)
                .map(|i| noise(128, seed * 100 + i, scale).try_into().unwrap())
                .collect();
            for head in 0..16 {
                let mut out = [0.0f64; 64];
                window64(&ring, head, &mut out);
                let g = &*PROTO_64;
                for (t, &o) in out.iter().enumerate() {
                    let mut acc = 0.0;
                    for i in 0..16 {
                        let vi = &ring[(head + i) % 16];
                        let n = t + 64 * i;
                        let val = if (n / 128) % 2 == 1 {
                            -vi[n % 128]
                        } else {
                            vi[n % 128]
                        };
                        acc += g[n] * val;
                    }
                    assert_eq!(
                        o.to_bits(),
                        f64::to_bits(acc),
                        "window64 seed {seed} head {head} t {t}"
                    );
                }
            }
        }
    }

    /// The two prototypes are low-pass filters of a 32-band bank: a constant
    /// on subband 0 must come out as that constant, once the structure is
    /// scaled by [`RECONSTRUCTION_GAIN`]. This pins the overall gain of the
    /// implementation and would fail for a shifted or transposed table.
    #[test]
    fn dc_on_subband_zero_comes_out_as_dc() {
        for perfect in [false, true] {
            let mut q = Qmf::new();
            let mut xin = [0.0f64; 32];
            xin[0] = 1000.0;
            let mut out = [0.0f64; 32];
            let mut last = Vec::new();
            for _ in 0..40 {
                q.synthesize(&xin, perfect, &mut out);
                last = out.to_vec();
            }
            let mean = last.iter().sum::<f64>() / 32.0;
            let ripple = last.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max);
            assert!(
                (mean - 1000.0).abs() < 1.0,
                "perfect={perfect}: DC gain should be unity on the subband scale, got mean {mean}"
            );
            // A lone constant on subband 0 is not what the analysis bank
            // would produce for DC PCM (the aliasing terms in the other
            // subbands are missing), so the output is not perfectly flat:
            // the non-perfect prototype ripples < 0.1 %, the perfect one
            // 0.33 %. A shifted or transposed table gives tens of percent.
            assert!(
                ripple < 10.0,
                "perfect={perfect}: ripple {ripple} on a DC input"
            );
        }
    }

    /// Silence in, silence out — and the accumulator carry does not leak.
    #[test]
    fn zero_input_is_exactly_zero() {
        let mut q = Qmf::new();
        let xin = [0.0f64; 32];
        let mut out = [1.0f64; 32];
        for _ in 0..20 {
            q.synthesize(&xin, true, &mut out);
        }
        assert!(out.iter().all(|v| *v == 0.0));
    }

    /// A constant decimated LFE stream interpolates to that constant: the
    /// polyphase sums of both FIRs are unity (the 64× taps sum to 64).
    #[test]
    fn lfe_dc_gain_is_unity() {
        for factor in [64usize, 128] {
            let mut l = LfeInterp::new();
            let mut out = Vec::new();
            l.interpolate(&[500.0; 16], factor, &mut out);
            assert_eq!(out.len(), 16 * factor);
            let tail = &out[out.len() - factor..];
            for v in tail {
                assert!((v - 500.0).abs() < 0.5, "factor {factor}: {v}");
            }
        }
    }

    /// The D.8 prototype with the signs of every second block of 64 taps
    /// negated back, as `QMFInterpolation` uses it implicitly.
    fn prototype_32(perfect: bool) -> Vec<f64> {
        let c: &[f32; 512] = if perfect {
            &tables::QMF_FIR_PERFECT
        } else {
            &tables::QMF_FIR_NON_PERFECT
        };
        c.iter()
            .enumerate()
            .map(|(n, &v)| {
                if (n / 64) % 2 == 1 {
                    -(v as f64)
                } else {
                    v as f64
                }
            })
            .collect()
    }

    /// Spec-derived float reference: C.3.6 is the cosine-modulated bank
    /// `f_k[n] = 64 · s_k · g[n] · cos(π/32 · (k+½) · (n+16.5))`. Every
    /// subband's impulse response through [`Qmf`] must equal it, for both
    /// prototypes, and a random input must equal the direct convolution.
    #[test]
    fn c36_is_the_cosine_modulated_bank() {
        use std::f64::consts::PI;
        for perfect in [false, true] {
            let g = prototype_32(perfect);
            let f = |k: usize, n: usize| {
                let s = if ((2 * k + 1) as f64 * PI / 4.0).cos() > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                64.0 * s * g[n] * (PI / 32.0 * (k as f64 + 0.5) * (n as f64 + 16.5)).cos()
            };
            for k in 0..32 {
                let mut q = Qmf::new();
                let mut out = [0.0; 32];
                for m in 0..17 {
                    let mut xin = [0.0; 32];
                    if m == 0 {
                        xin[k] = 1.0;
                    }
                    q.synthesize(&xin, perfect, &mut out);
                    for (t, y) in out.iter().enumerate() {
                        let n = 32 * m + t;
                        let want = if n < 512 { f(k, n) } else { 0.0 };
                        assert!(
                            (y - want).abs() < 1e-9,
                            "perfect={perfect} k={k} n={n}: {y} vs {want}"
                        );
                    }
                }
            }
            // Random subband input against the direct convolution.
            let mut seed = 12345u64;
            let mut rnd = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            };
            let x: Vec<[f64; 32]> = (0..24).map(|_| std::array::from_fn(|_| rnd())).collect();
            let mut q = Qmf::new();
            let mut got = Vec::new();
            for xm in &x {
                let mut out = [0.0; 32];
                q.synthesize(xm, perfect, &mut out);
                got.extend_from_slice(&out);
            }
            for (n, y) in got.iter().enumerate() {
                let mut want = 0.0;
                for (m, xm) in x.iter().enumerate() {
                    if n >= 32 * m && n - 32 * m < 512 {
                        want += (0..32).map(|k| xm[k] * f(k, n - 32 * m)).sum::<f64>();
                    }
                }
                assert!((y - want).abs() < 1e-9, "perfect={perfect} n={n}");
            }
        }
    }

    /// The 64-band bank against its float reference (direct convolution of
    /// the D.9 prototype modulated as documented on [`Qmf64`]).
    #[test]
    fn qmf64_is_the_direct_form() {
        use std::f64::consts::PI;
        let g: Vec<f64> = tables::X96_QMF_FIR
            .iter()
            .enumerate()
            .map(|(n, &v)| if (n / 128) % 2 == 1 { -v } else { v })
            .collect();
        let f = |k: usize, n: usize| {
            let s = if ((2 * k + 1) as f64 * PI / 4.0).cos() > 0.0 {
                1.0
            } else {
                -1.0
            };
            128.0 * s * g[n] * (PI / 64.0 * (k as f64 + 0.5) * (n as f64 + 32.5)).cos()
        };
        let x: Vec<[f64; 64]> = (0..20)
            .map(|m| {
                std::array::from_fn(|k| {
                    (((m * 64 + k) * 2654435761usize) % 1000) as f64 / 500.0 - 1.0
                })
            })
            .collect();
        let mut q = Qmf64::new();
        let mut got = Vec::new();
        for xm in &x {
            let mut out = [0.0; 64];
            q.synthesize(xm, &mut out);
            got.extend_from_slice(&out);
        }
        for (n, y) in got.iter().enumerate() {
            let mut want = 0.0;
            for (m, xm) in x.iter().enumerate() {
                if n >= 64 * m && n - 64 * m < 1024 {
                    want += (0..64).map(|k| xm[k] * f(k, n - 64 * m)).sum::<f64>();
                }
            }
            assert!((y - want).abs() < 1e-9, "n={n}: {y} vs {want}");
        }
    }

    /// 6.2.3: core subband samples in the lower 32 bands of the 64-band bank
    /// (upper bands zero) give the 32-band bank's output interpolated to the
    /// doubled rate. A slowly varying signal in one subband: 64-band output
    /// sample j must match the 32-band output band-limited-interpolated at
    /// core time (j − ½)/2 (the pair is offset by half a high-rate sample,
    /// the difference of the prototypes' group delays).
    #[test]
    fn x96_bank_interpolates_the_core_bank() {
        use std::f64::consts::PI;
        for k in [0usize, 3, 10, 16, 22] {
            let (mut q32, mut q64) = (Qmf::new(), Qmf64::new());
            let (mut y32, mut y64) = (Vec::new(), Vec::new());
            for m in 0..80 {
                let v = (m as f64 * 0.3).sin() * 1000.0;
                let mut x32 = [0.0; 32];
                x32[k] = v;
                let mut x64 = [0.0; 64];
                x64[k] = v;
                let mut o32 = [0.0; 32];
                let mut o64 = [0.0; 64];
                q32.synthesize(&x32, false, &mut o32);
                q64.synthesize(&x64, &mut o64);
                y32.extend_from_slice(&o32);
                y64.extend_from_slice(&o64);
            }
            // Hann-windowed sinc interpolation of the core output, ±128 taps.
            let interp = |t: f64| {
                let c = t.floor() as isize;
                (c - 128..=c + 129)
                    .map(|m| {
                        let d = t - m as f64;
                        let sinc = if d.abs() < 1e-12 {
                            1.0
                        } else {
                            (PI * d).sin() / (PI * d)
                        };
                        let w = 0.5 + 0.5 * (PI * d / 130.0).cos();
                        y32[m as usize] * sinc * w
                    })
                    .sum::<f64>()
            };
            let (mut err, mut sig) = (0.0, 0.0);
            for j in 1200..3600 {
                let want = interp((j as f64 - 0.5) / 2.0);
                err += (y64[j] - want).powi(2);
                sig += want.powi(2);
            }
            let snr = 10.0 * (sig / err).log10();
            assert!(
                snr > 40.0,
                "band {k}: 64-band vs interpolated 32-band SNR {snr:.1} dB"
            );
        }
    }

    #[test]
    fn lfe_2x_dc_gain_is_unity() {
        let mut l = Lfe2x::default();
        let mut out = Vec::new();
        l.interpolate(&[100.0; 8], &mut out);
        for v in &out[4..] {
            assert!((v - 100.0).abs() < 0.01, "{v}");
        }
    }
}
