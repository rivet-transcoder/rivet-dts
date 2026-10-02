//! Inverse ADPCM of the subband samples (Annex C.3.3) and the per-subband
//! history it runs against.
//!
//! A predicted subband carries residuals; the decoder restores each sample
//! as `x[m] = r[m] + Σ_{n<4} c[n]·x[m−n−1]`, where `x[m−1] … x[m−4]` are the
//! subband's previously reconstructed samples — earlier in this subsubframe,
//! subframe or frame, or (when `HFLAG` = 1) the previous frame. The
//! coefficients come from the D.10.1 code book, which ETSI does not print
//! (see [`crate::vq`]); without it the decoder either refuses the frame or,
//! when asked, estimates the predictor from the history ([`AdpcmFallback`]).

/// Order of the core predictor (`NumADPCMCoeff`).
pub(crate) const ORDER: usize = 4;
/// Reconstructed samples kept per subband across frames: 4 for the
/// predictor, more for the [`AdpcmFallback::Estimate`] analysis.
pub(crate) const HIST: usize = 32;

/// What to do with a predicted subband (`PMODE` = 1) when no D.10.1 code
/// book has been given to the decoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdpcmFallback {
    /// Refuse the frame with [`Error::Unsupported`](crate::Error::Unsupported)
    /// naming the missing code book (the default): nothing is output that
    /// the stream does not determine.
    #[default]
    Refuse,
    /// Decode the frame, replacing each transmitted predictor by one
    /// estimated from the subband's own reconstructed history (4th-order
    /// autocorrelation LPC over the last 32 samples, bandwidth-expanded).
    /// ADPCM is only used by encoders on strongly predictable (tonal)
    /// subbands, where such an estimate is close; the result is an
    /// approximation, not the decoded stream, and [`Decoder::adpcm_estimated`](crate::Decoder::adpcm_estimated)
    /// counts the subbands it was used on.
    Estimate,
}

/// One channel's reconstructed subband history.
#[derive(Clone)]
pub(crate) struct PredictorState {
    /// `carry[sb]`: the last [`HIST`] samples of subband `sb` from previous
    /// frames, oldest first.
    carry: Vec<[f64; HIST]>,
    /// Whether this frame predicts from the carried history (`HFLAG`).
    use_carry: bool,
    /// Subbands whose predictor was estimated, for the decoder's count.
    pub estimated: u64,
}

impl PredictorState {
    pub fn new(bands: usize) -> Self {
        Self { carry: vec![[0.0; HIST]; bands], use_carry: false, estimated: 0 }
    }

    /// Start a frame: `hflag` says whether the previous frame's history is
    /// used.
    pub fn begin_frame(&mut self, hflag: bool) {
        self.use_carry = hflag;
    }

    /// Sample `k` of subband `sb` relative to the frame start, where
    /// negative `k` reaches into the carried history.
    fn at(&self, sb: usize, frame: &[f64], k: isize) -> f64 {
        if k >= 0 {
            frame[k as usize]
        } else if self.use_carry && -k <= HIST as isize {
            self.carry[sb][(HIST as isize + k) as usize]
        } else {
            0.0
        }
    }

    /// Annex C.3.3 for the 8 samples at `t..t + 8` of `frame` (which holds
    /// the subband's samples from the frame start, residuals at `t..`).
    pub fn inverse(&self, sb: usize, c: &[f64; ORDER], frame: &mut [f64], t: usize) {
        for m in t..frame.len() {
            let mut acc = frame[m];
            for (n, cn) in c.iter().enumerate() {
                acc += cn * self.at(sb, frame, m as isize - n as isize - 1);
            }
            frame[m] = acc;
        }
    }

    /// Estimate a predictor for subband `sb` from its reconstructed samples
    /// before this subsubframe (`prior`, from the frame start) and the
    /// carried history ([`AdpcmFallback::Estimate`]).
    pub fn estimate(&mut self, sb: usize, prior: &[f64]) -> [f64; ORDER] {
        self.estimated += 1;
        let t = prior.len() as isize;
        let window: Vec<f64> = (t - HIST as isize..t).map(|k| self.at(sb, prior, k)).collect();
        lpc(&window)
    }

    /// End a frame: carry the last [`HIST`] samples of every subband.
    pub fn end_frame(&mut self, bands: &[Vec<f64>]) {
        for (sb, s) in bands.iter().enumerate().take(self.carry.len()) {
            let n = s.len();
            let mut next = [0.0; HIST];
            for (i, v) in next.iter_mut().enumerate() {
                let k = n as isize - HIST as isize + i as isize;
                *v = if k >= 0 {
                    s[k as usize]
                } else if self.use_carry {
                    self.carry[sb][(HIST as isize + k) as usize]
                } else {
                    0.0
                };
            }
            self.carry[sb] = next;
        }
    }
}

/// 4th-order LPC by the autocorrelation method (Levinson–Durbin) with a
/// small white-noise correction and 0.98 bandwidth expansion, in the
/// predictor sign of C.3.3 (`x[m] ≈ Σ c[n]·x[m−n−1]`). Silent or
/// unpredictable history gives no prediction.
pub(crate) fn lpc(x: &[f64]) -> [f64; ORDER] {
    let mut r = [0.0f64; ORDER + 1];
    for (lag, rl) in r.iter_mut().enumerate() {
        *rl = x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum();
    }
    if r[0] <= 1e-9 {
        return [0.0; ORDER];
    }
    r[0] *= 1.0 + 1e-4;
    let mut a = [0.0f64; ORDER + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        if k.abs() >= 1.0 {
            break;
        }
        let prev = a;
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
    }
    let mut g = 1.0;
    std::array::from_fn(|n| {
        g *= 0.98;
        -a[n + 1] * g
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Annex C.3.3 written as plainly as the annex prints it, over a whole
    /// sequence with the history as negative indices.
    fn reference(residual: &[f64], history: &[f64; 4], c: &[f64; 4]) -> Vec<f64> {
        // raSample[-4..-1] = history (oldest first), then the samples.
        let mut ra: Vec<f64> = history.to_vec();
        ra.extend_from_slice(residual);
        for m in 4..ra.len() {
            for n in 0..4 {
                ra[m] += c[n] * ra[m - n - 1];
            }
        }
        ra[4..].to_vec()
    }

    #[test]
    fn inverse_adpcm_matches_the_annex_c_3_3_reference() {
        let c = [1.2119140625, -0.6, 0.25, -0.0625];
        let mut st = PredictorState::new(32);
        // Previous frame ends with these samples in subband 3.
        let mut prev = vec![vec![0.0; 16]; 32];
        prev[3][12..].copy_from_slice(&[0.5, -1.0, 2.0, 3.0]);
        st.begin_frame(true);
        st.end_frame(&prev);
        st.begin_frame(true);
        let residual: Vec<f64> = (0..16).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.1).collect();
        let mut frame = residual.clone();
        // Two subsubframes, predicted one after the other.
        st.inverse(3, &c, &mut frame[..8], 0);
        st.inverse(3, &c, &mut frame[..16], 8);
        let want = reference(&residual, &[0.5, -1.0, 2.0, 3.0], &c);
        for (a, b) in frame.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        // HFLAG = 0: the previous frame's history is ignored.
        let mut st2 = st.clone();
        st2.begin_frame(false);
        let mut frame = residual.clone();
        st2.inverse(3, &c, &mut frame[..8], 0);
        let want = reference(&residual[..8], &[0.0; 4], &c);
        assert!(frame[..8].iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn lpc_recovers_a_resonator() {
        // x[m] = 2 r cosθ x[m−1] − r² x[m−2]: a damped sinusoid.
        let (r, th) = (0.99f64, 0.7f64);
        let mut x = vec![1.0, 2.0 * r * th.cos()];
        for m in 2..32 {
            x.push(2.0 * r * th.cos() * x[m - 1] - r * r * x[m - 2]);
        }
        let c = lpc(&x);
        // Predict the last 8 from their history: error well under the signal.
        let (mut e, mut s) = (0.0, 0.0);
        for m in 24..32 {
            let p: f64 = (0..4).map(|n| c[n] * x[m - n - 1]).sum();
            e += (x[m] - p).powi(2);
            s += x[m].powi(2);
        }
        assert!(e < s * 1e-2, "prediction gain too low: {}", s / e);
        assert_eq!(lpc(&[0.0; 32]), [0.0; 4]);
    }
}
