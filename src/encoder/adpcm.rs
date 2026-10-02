//! The encoder's side of the core ADPCM (Annex C.3.3): choosing a
//! prediction vector from the D.10.1 code book for one subband subframe.
//!
//! The decoder reconstructs `x[m] = r[m] + Σ_{n<4} c[n]·x[m−n−1]`, where the
//! history `x[−1..−4]` is the subband's previous reconstruction. For a
//! candidate `c` the open-loop prediction error over the subframe is the
//! quadratic form `E(c) = E₀ − 2·cᵀp + cᵀRc`, with `R` and `p` the
//! covariances of the regressors; every one of the 4 096 vectors is scored
//! that way and the best is kept when its prediction gain pays for the
//! 12-bit `PVQ`. The residual itself is then quantised closed-loop by the
//! caller, against the history the decoder will have.

use crate::vq::{ADPCM_ORDER, ADPCM_VECTORS, AdpcmCodebook};

/// Smallest prediction gain (linear power ratio) worth signalling: 16
/// samples at ½·log₂(G) bits each must save more than the 12-bit `PVQ` and
/// the `PMODE` flag, with some margin for the coarser quantiser steps.
pub const MIN_GAIN: f64 = 5.0;

/// Covariances of the 4th-order prediction problem over one subframe.
struct Normal {
    e0: f64,
    p: [f64; ADPCM_ORDER],
    r: [[f64; ADPCM_ORDER]; ADPCM_ORDER],
}

/// `hist[n]` is `x[−n−1]` (newest first), `x` the subframe's samples.
fn normal(hist: &[f64; ADPCM_ORDER], x: &[f64]) -> Normal {
    let mut seq = Vec::with_capacity(ADPCM_ORDER + x.len());
    seq.extend(hist.iter().rev());
    seq.extend_from_slice(x);
    let mut nm = Normal { e0: 0.0, p: [0.0; ADPCM_ORDER], r: [[0.0; ADPCM_ORDER]; ADPCM_ORDER] };
    for (m, &xm) in x.iter().enumerate() {
        let reg: [f64; ADPCM_ORDER] = std::array::from_fn(|n| seq[m + ADPCM_ORDER - n - 1]);
        nm.e0 += xm * xm;
        for n in 0..ADPCM_ORDER {
            nm.p[n] += xm * reg[n];
            for l in 0..ADPCM_ORDER {
                nm.r[n][l] += reg[n] * reg[l];
            }
        }
    }
    nm
}

fn error(nm: &Normal, c: &[f64; ADPCM_ORDER]) -> f64 {
    let mut e = nm.e0;
    for n in 0..ADPCM_ORDER {
        e -= 2.0 * c[n] * nm.p[n];
        for l in 0..ADPCM_ORDER {
            e += c[n] * nm.r[n][l] * c[l];
        }
    }
    e
}

/// The code book vector with the largest open-loop prediction gain, if
/// that gain is at least [`MIN_GAIN`]: `(PVQ, gain)`.
pub fn best_vector(book: &AdpcmCodebook, hist: &[f64; ADPCM_ORDER], x: &[f64]) -> Option<(usize, f64)> {
    let nm = normal(hist, x);
    if nm.e0 <= 0.0 {
        return None;
    }
    let mut best = (0usize, f64::INFINITY);
    for pvq in 0..ADPCM_VECTORS {
        let e = error(&nm, &book.coefficients(pvq));
        if e < best.1 {
            best = (pvq, e);
        }
    }
    let gain = nm.e0 / best.1.max(nm.e0 * 1e-12);
    (gain >= MIN_GAIN).then_some((best.0, gain))
}

#[cfg(test)]
/// Annex C.3.3 inverse ADPCM, written as the reference: `r` residuals in,
/// reconstruction out, `hist` updated (newest first).
pub fn inverse_adpcm(c: &[f64; ADPCM_ORDER], hist: &mut [f64; ADPCM_ORDER], r: &[f64]) -> Vec<f64> {
    r.iter()
        .map(|&rm| {
            let x = rm + (0..ADPCM_ORDER).map(|n| c[n] * hist[n]).sum::<f64>();
            hist.rotate_right(1);
            hist[0] = x;
            x
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tonal subband signal is predicted with high gain from the test
    /// book, and the quadratic-form error equals the directly computed one.
    #[test]
    fn finds_a_predictor_for_a_tone_and_scores_it_exactly() {
        let book = AdpcmCodebook::private_test_book();
        let sig: Vec<f64> = (0..20).map(|n| 1000.0 * (0.9 * n as f64 + 0.3).cos()).collect();
        let hist = [sig[3], sig[2], sig[1], sig[0]];
        let x = &sig[4..];
        let (pvq, gain) = best_vector(&book, &hist, x).expect("a tone is predictable");
        assert!(gain > 100.0, "gain {gain}");
        let c = book.coefficients(pvq);
        let mut h = hist;
        let direct: f64 = x
            .iter()
            .map(|&xm| {
                let p: f64 = (0..4).map(|n| c[n] * h[n]).sum();
                h.rotate_right(1);
                h[0] = xm;
                (xm - p).powi(2)
            })
            .sum();
        let nm = normal(&hist, x);
        assert!((error(&nm, &c) - direct).abs() < 1e-6 * direct.max(1.0));
        // White noise is not predictable.
        let noise: Vec<f64> = (0..20).map(|n| ((n * 7919 % 101) as f64 - 50.0) * 10.0).collect();
        assert!(best_vector(&book, &[noise[3], noise[2], noise[1], noise[0]], &noise[4..]).is_none());
    }
}
