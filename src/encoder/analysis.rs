//! The encoder's filter banks: a 32-band analysis bank matched to the
//! decoder's synthesis bank (Annex C.3.6), and the LFE decimator matched to
//! the 64× interpolation FIR (Annex C.3.7).
//!
//! ETSI TS 102 114 describes the encoder only in outline (Figure 5-1: "split
//! and decimated by a 32-band QMF bank"), so the analysis bank is derived
//! from the synthesis bank the specification does define. The synthesis
//! bank is linear and periodically time-varying with period 32: a unit
//! sample on subband `k` in block `m` produces the output `g_k[n − 32m]`.
//! The analysis taken here is the matched one,
//!
//! ```text
//! s_k[m] = c · Σ_n x[n] · g_k[n − 32m],
//! ```
//!
//! which inverts the synthesis exactly when the 32·(shifts) functions
//! `g_k[n − 32m]` form a tight frame — what a perfect-reconstruction
//! cosine-modulated bank is. `c` is `1 / A` with `A` the frame bound,
//! measured from the `g_k` themselves. With the `FILTS` = 1 prototype the
//! unquantised round trip reconstructs to better than 100 dB SNR; the
//! `FILTS` = 0 prototype is only near-perfect (see the tests).
//!
//! The matched analysis looks `L` samples ahead of the block it codes, so
//! the encoder prepends [`DELAY`] zeros: decoded sample `n` reconstructs
//! input sample `n − DELAY`. The LFE path is delayed by the same amount.

use std::sync::LazyLock;

use crate::tables;

/// Subbands of the core filter bank.
pub const BANDS: usize = 32;
/// Number of output blocks the impulse responses are recorded over: the
/// prototype has 512 taps, so 17 blocks of 32 hold every non-zero sample.
const IR_BLOCKS: usize = 18;
/// Length of every recorded synthesis function.
pub const IR_LEN: usize = IR_BLOCKS * 32;

/// End-to-end delay of encoder + decoder in samples at the core rate:
/// the lookahead of the matched analysis, a whole number of blocks.
pub const DELAY: usize = 512;

/// `RECONSTRUCTION_GAIN` of the decoder's synthesis (`synth.rs`).
const RECONSTRUCTION_GAIN: f64 = 128.0 * std::f64::consts::SQRT_2;

/// One synthesis call of Annex C.3.6, as the decoder runs it, on private
/// state: used only to record the bank's impulse responses.
struct RefSynth {
    x: [f64; 512],
    z: [f64; 64],
}

impl RefSynth {
    fn new() -> Self {
        Self {
            x: [0.0; 512],
            z: [0.0; 64],
        }
    }

    fn run(&mut self, xin: &[f64; BANDS], perfect: bool, out: &mut [f64; 32]) {
        use std::f64::consts::PI;
        let coeff: &[f32; 512] = if perfect {
            &tables::QMF_FIR_PERFECT
        } else {
            &tables::QMF_FIR_NON_PERFECT
        };
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        for (k, ak) in a.iter_mut().enumerate() {
            for i in 0..16 {
                *ak += (xin[2 * i] + xin[2 * i + 1])
                    * (((2 * i + 1) * (2 * k + 1)) as f64 * PI / 64.0).cos();
            }
        }
        for (k, bk) in b.iter_mut().enumerate() {
            for i in 0..16 {
                let v = if i > 0 {
                    xin[2 * i] + xin[2 * i - 1]
                } else {
                    xin[0]
                };
                *bk += v * ((i * (2 * k + 1)) as f64 * PI / 32.0).cos();
            }
        }
        for k in 0..16 {
            let c = 0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).cos());
            let s = -0.25 / (2.0 * ((2 * k + 1) as f64 * PI / 128.0).sin());
            self.x[k] = c * (a[k] + b[k]);
            self.x[32 - k - 1] = s * (a[k] - b[k]);
        }
        for i in 0..32 {
            let k = 31 - i;
            let (mut acc, mut acc2) = (0.0f64, 0.0f64);
            for j in (0..512).step_by(64) {
                acc += coeff[i + j] as f64 * (self.x[i + j] - self.x[j + k]);
                acc2 += coeff[32 + i + j] as f64 * (-self.x[i + j] - self.x[j + k]);
            }
            self.z[i] += acc;
            self.z[32 + i] += acc2;
        }
        for (o, z) in out.iter_mut().zip(&self.z[..32]) {
            *o = z * RECONSTRUCTION_GAIN;
        }
        self.x.copy_within(0..480, 32);
        self.z.copy_within(32..64, 0);
        self.z[32..].fill(0.0);
    }
}

/// The matched analysis bank for one prototype.
pub struct AnalysisBank {
    /// `g[k][n]`, pre-multiplied by `c`, stored by `n` (all bands' taps
    /// for one input sample together).
    g: Vec<[f64; BANDS]>,
    /// Mean energy of the (unscaled) synthesis functions, `Σ_n g_k[n]²`:
    /// the factor from subband-domain noise power to output noise energy.
    pub synthesis_energy: f64,
}

impl AnalysisBank {
    fn build(perfect: bool) -> Self {
        let mut g = vec![[0.0f64; IR_LEN]; BANDS];
        for (k, gk) in g.iter_mut().enumerate() {
            let mut s = RefSynth::new();
            let mut out = [0.0f64; 32];
            for blk in 0..IR_BLOCKS {
                let mut xin = [0.0f64; BANDS];
                if blk == 0 {
                    xin[k] = 1.0;
                }
                s.run(&xin, perfect, &mut out);
                gk[blk * 32..blk * 32 + 32].copy_from_slice(&out);
            }
        }
        // Frame bound: Σ_k Σ_m g_k[n − 32m]² is 32-periodic in n; average it.
        let mut a = 0.0;
        for n in 0..32 {
            for gk in &g {
                for m in 0..IR_BLOCKS {
                    a += gk[n + 32 * m].powi(2);
                }
            }
        }
        a /= 32.0;
        let synthesis_energy = g
            .iter()
            .map(|gk| gk.iter().map(|v| v * v).sum::<f64>())
            .sum::<f64>()
            / BANDS as f64;
        for gk in &mut g {
            for v in gk.iter_mut() {
                *v /= a;
            }
        }
        let g = (0..IR_LEN)
            .map(|n| std::array::from_fn(|k| g[k][n]))
            .collect();
        Self {
            g,
            synthesis_energy,
        }
    }

    /// The bank for `FILTS` = 1 (`perfect`) or 0.
    pub fn get(perfect: bool) -> &'static AnalysisBank {
        static PR: LazyLock<AnalysisBank> = LazyLock::new(|| AnalysisBank::build(true));
        static NPR: LazyLock<AnalysisBank> = LazyLock::new(|| AnalysisBank::build(false));
        if perfect { &PR } else { &NPR }
    }

    /// One block of subband samples: `x` holds `IR_LEN` input samples
    /// starting at the block's first sample.
    pub fn analyse(&self, x: &[f64], out: &mut [f64; BANDS]) {
        analyse(&self.g, &x[..IR_LEN], out);
    }
}

crate::simd::avx2_or_portable! {
    /// `out[k] = Σ_n g_k[n] x[n]`, each band's sum in order of `n` (from
    /// -0.0, as `Iterator::sum` starts) but all bands at once.
    fn analyse(g: &[[f64; BANDS]], x: &[f64], out: &mut [f64; BANDS]) {
        let mut acc = [-0.0f64; BANDS];
        for (gn, &xn) in g.iter().zip(x) {
            for k in 0..BANDS {
                acc[k] += gn[k] * xn;
            }
        }
        *out = acc;
    }
}

/// Samples of input the LFE decimator reads per decimated sample.
pub const LFE_TAPS: usize = 512;
/// LFE decimation factor (`LFF` = 2).
pub const LFE_FACTOR: usize = 64;

/// Upper edge of the LFE passband the decimator equalises, in Hz (at most
/// 0.45 of the decimated Nyquist frequency, so 112 Hz at 32 kHz).
pub const LFE_PASSBAND_HZ: f64 = 140.0;

/// The LFE decimation filter for one sample rate: 512 taps on the same
/// support as the decoder's 64× interpolation FIR `h` (D.8), so decoded
/// LFE sample `n` lines up with input sample `n` as the main channels do.
///
/// The interpolator is a gentle low-pass (−0.6 dB at 40 Hz, −5.7 dB at
/// 120 Hz at 48 kHz), so the plain matched decimator (`h / 64`) would square
/// that droop. This one is designed by weighted least squares instead:
/// symmetric like `h`, the cascade `A(f)·H(f)/64` flat to
/// [`LFE_PASSBAND_HZ`], and `A(f)` small above the decimated Nyquist
/// frequency (`fs/128`), where whatever passes aliases.
pub struct LfeDecimator {
    taps: [f64; LFE_TAPS],
}

impl LfeDecimator {
    fn design(sample_rate: f64) -> Self {
        use std::f64::consts::PI;
        const HALF: usize = LFE_TAPS / 2;
        let h = &tables::LFE_FIR_64X;
        // Zero-phase amplitude of a symmetric 512-tap filter from its upper half.
        let basis = |f: f64, k: usize| 2.0 * (2.0 * PI * f / sample_rate * (k as f64 + 0.5)).cos();
        let h_amp = |f: f64| {
            (0..HALF)
                .map(|k| h[HALF + k] as f64 * basis(f, k))
                .sum::<f64>()
        };
        let nyq = sample_rate / 128.0;
        let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut f = 0.0;
        while f <= LFE_PASSBAND_HZ.min(0.45 * nyq) {
            let g = h_amp(f) / LFE_FACTOR as f64;
            rows.push((
                (0..HALF).map(|k| PASS_WEIGHT * basis(f, k) * g).collect(),
                PASS_WEIGHT,
            ));
            f += 1.0;
        }
        const PASS_WEIGHT: f64 = 4.0;
        let stop_weight = 30.0f64;
        let mut f = nyq;
        while f < sample_rate / 2.0 {
            // Weighted by where it lands: input at f folds to fa and is
            // heard at the interpolator's gain there.
            let mut fa = f % (2.0 * nyq);
            if fa > nyq {
                fa = 2.0 * nyq - fa;
            }
            let wt = stop_weight * (h_amp(fa).abs() / LFE_FACTOR as f64).max(1e-3);
            rows.push(((0..HALF).map(|k| wt * basis(f, k)).collect(), 0.0));
            f += if f < 4.0 * nyq { 4.0 } else { 50.0 };
        }
        // Normal equations, with a small ridge.
        let mut m = vec![vec![0.0f64; HALF + 1]; HALF];
        for (r, t) in &rows {
            for i in 0..HALF {
                for j in 0..HALF {
                    m[i][j] += r[i] * r[j];
                }
                m[i][HALF] += r[i] * t;
            }
        }
        let trace: f64 = (0..HALF).map(|i| m[i][i]).sum();
        for (i, row) in m.iter_mut().enumerate() {
            row[i] += 1e-10 * trace / HALF as f64;
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..HALF {
            let piv = (col..HALF)
                .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
                .expect("rows");
            m.swap(col, piv);
            for r in col + 1..HALF {
                let fac = m[r][col] / m[col][col];
                if fac != 0.0 {
                    for c in col..=HALF {
                        m[r][c] -= fac * m[col][c];
                    }
                }
            }
        }
        let mut b = vec![0.0f64; HALF];
        for i in (0..HALF).rev() {
            let s: f64 = (i + 1..HALF).map(|j| m[i][j] * b[j]).sum();
            b[i] = (m[i][HALF] - s) / m[i][i];
        }
        let mut taps = [0.0f64; LFE_TAPS];
        for k in 0..HALF {
            taps[HALF + k] = b[k];
            taps[HALF - 1 - k] = b[k];
        }
        Self { taps }
    }

    /// The decimator for `sample_rate` (48 000, 44 100 or 32 000).
    pub fn get(sample_rate: u32) -> &'static LfeDecimator {
        static R48: LazyLock<LfeDecimator> = LazyLock::new(|| LfeDecimator::design(48_000.0));
        static R441: LazyLock<LfeDecimator> = LazyLock::new(|| LfeDecimator::design(44_100.0));
        static R32: LazyLock<LfeDecimator> = LazyLock::new(|| LfeDecimator::design(32_000.0));
        match sample_rate {
            44_100 => &R441,
            32_000 => &R32,
            _ => &R48,
        }
    }

    /// One decimated LFE sample: `x` holds [`LFE_TAPS`] input samples
    /// starting at `64·j`.
    pub fn decimate(&self, x: &[f64]) -> f64 {
        self.taps.iter().zip(x).map(|(a, v)| a * v).sum()
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

    /// The bank's all-bands-at-once sums equal each band's own `Iterator::sum`
    /// of products, bit for bit.
    #[test]
    fn analysis_matches_per_band_sums() {
        for perfect in [false, true] {
            let bank = AnalysisBank::get(perfect);
            for (seed, scale) in [(1u64, 1.0), (2, 8_388_608.0), (3, 1e-300), (4, 0.0)] {
                let x = noise(IR_LEN + 5, seed, scale);
                let mut out = [0.0f64; BANDS];
                bank.analyse(&x, &mut out);
                for (k, &o) in out.iter().enumerate() {
                    let want: f64 = (0..IR_LEN).map(|n| bank.g[n][k] * x[n]).sum();
                    assert_eq!(o.to_bits(), want.to_bits(), "band {k} seed {seed}");
                }
            }
        }
    }
    use crate::synth::Qmf;

    /// The LFE decimator against the decoder's interpolator: flat within
    /// ±0.3 dB to 120 Hz (100 Hz at 32 kHz), and input above fs/128 reaches the
    /// output (folded, through the interpolator) at least 50 dB down (35 dB at
    /// 32 kHz).
    #[test]
    fn lfe_decimator_equalises_the_interpolator() {
        for rate in [48_000u32, 44_100, 32_000] {
            let a = LfeDecimator::get(rate);
            let resp = |taps: &[f64], f: f64| {
                let w = 2.0 * std::f64::consts::PI * f / rate as f64;
                let (re, im) = taps
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(re, im), (n, v)| {
                        (re + v * (w * n as f64).cos(), im + v * (w * n as f64).sin())
                    });
                (re * re + im * im).sqrt()
            };
            let h: Vec<f64> = tables::LFE_FIR_64X.iter().map(|v| *v as f64).collect();
            let mut f = 0.0;
            while f <= 120.0f64.min(0.4 * rate as f64 / 128.0) {
                let g = 20.0 * (resp(&a.taps, f) * resp(&h, f) / 64.0).log10();
                assert!(
                    g.abs() < 0.3,
                    "{rate} Hz: cascade gain {g:+.2} dB at {f} Hz"
                );
                f += 5.0;
            }
            // Input above fs/128 folds to `fa` and comes out at the
            // interpolator's gain there: what reaches the output.
            let nyq = rate as f64 / 128.0;
            let mut worst = f64::MIN;
            let mut f = nyq;
            while f < rate as f64 / 2.0 {
                let mut fa = f % (2.0 * nyq);
                if fa > nyq {
                    fa = 2.0 * nyq - fa;
                }
                worst = worst.max(20.0 * (resp(&a.taps, f) * resp(&h, fa) / 64.0).log10());
                f += 3.0;
            }
            eprintln!("LFE decimator at {rate} Hz: worst alias reaching the output {worst:.1} dB");
            // 32 kHz leaves the narrowest transition band (112 → 250 Hz).
            let floor = if rate == 32_000 { -35.0 } else { -50.0 };
            assert!(worst < floor, "{rate} Hz: aliasing only {worst:.1} dB down");
        }
    }

    /// The private copy of the synthesis bank agrees with the decoder's.
    #[test]
    fn reference_synthesis_matches_the_decoder() {
        for perfect in [false, true] {
            let mut a = RefSynth::new();
            let mut b = Qmf::new();
            let (mut oa, mut ob) = ([0.0; 32], [0.0; 32]);
            for blk in 0..40 {
                let xin: [f64; 32] =
                    std::array::from_fn(|k| ((blk * 31 + k * 7) % 13) as f64 - 6.0);
                a.run(&xin, perfect, &mut oa);
                b.synthesize(&xin, perfect, &mut ob);
                assert_eq!(oa, ob);
            }
        }
    }

    /// Analysis then the decoder's synthesis, unquantised, reconstructs the
    /// input delayed by [`DELAY`]: > 100 dB for the perfect-reconstruction
    /// prototype; the non-perfect one is reported.
    #[test]
    fn unquantised_round_trip_reconstructs() {
        for perfect in [true, false] {
            let bank = AnalysisBank::get(perfect);
            let n = 32 * 200;
            let input: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    1e6 * (t * 0.0123).sin()
                        + 5e5 * (t * 0.71).sin()
                        + 2e5 * ((i * 7919 % 1000) as f64 / 500.0 - 1.0)
                })
                .collect();
            let mut padded = vec![0.0; DELAY];
            padded.extend_from_slice(&input);
            padded.extend(std::iter::repeat_n(0.0, IR_LEN + DELAY));
            let mut q = Qmf::new();
            let mut out = Vec::new();
            let mut sb = [0.0; 32];
            let mut pcm = [0.0; 32];
            let blocks = (n + DELAY) / 32;
            for m in 0..blocks {
                bank.analyse(&padded[32 * m..], &mut sb);
                q.synthesize(&sb, perfect, &mut pcm);
                out.extend_from_slice(&pcm);
            }
            // Skip the first and last few blocks of the comparison.
            let (mut se, mut ss) = (0.0, 0.0);
            for i in 1024..n - 1024 {
                se += (out[i + DELAY] - input[i]).powi(2);
                ss += input[i].powi(2);
            }
            let snr = 10.0 * (ss / se).log10();
            eprintln!(
                "FILTS={}: unquantised round-trip SNR {snr:.1} dB",
                u8::from(perfect)
            );
            if perfect {
                assert!(snr > 100.0, "PR bank round trip {snr:.1} dB");
            } else {
                assert!(snr > 40.0, "NPR bank round trip {snr:.1} dB");
            }
        }
    }
}
