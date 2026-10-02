//! The two vector code books of the core that ETSI TS 102 114 describes but
//! does not print (Annex D.10: "Due to its extensive size, this table is not
//! included here"), as caller-supplied data.
//!
//! - [`AdpcmCodebook`] — D.10.1, the ADPCM prediction coefficients: 4 096
//!   vectors of four coefficients, each stored as the coefficient × 2¹³
//!   (the spec's own example: the first entry is 9 928, i.e. 1.211 914 062 5).
//!   Indexed by the 12-bit `PVQ` of the core, XCh, XXCh and X96 side
//!   information.
//! - [`HfVqCodebook`] — D.10.2, the high-frequency subband vectors: 1 024
//!   vectors of 32 elements, each element an 8-bit two's-complement integer
//!   divided by 2⁴. Indexed by the 10-bit `HFREQ` (core, XCh, XXCh) and
//!   `HFREQ96` (X96).
//!
//! This crate contains neither table, and never will take them from another
//! implementation. A caller holding a lawful copy (DTS licenses the
//! specification's full annex) can load it here and hand it to the
//! [`Decoder`](crate::Decoder) (and to the `Encoder`, so it
//! can predict). Without one, the decoder refuses predicted frames by name or
//! — if asked — estimates the predictor (see
//! `AdpcmFallback`), and decodes VQ subbands as
//! silence, which §5.4.3 allows.
//!
//! [`AdpcmCodebook::private_test_book`] builds a code book of stable
//! 4th-order predictors that is **not** D.10.1: streams predicted with it
//! decode correctly only on a decoder given the same book. It exists so the
//! prediction paths of the encoder and decoder can be exercised end to end.

use std::fmt;

use super::Error;

/// Number of vectors in the D.10.1 ADPCM code book (12-bit `PVQ`).
pub const ADPCM_VECTORS: usize = 4096;
/// Prediction order of the core ADPCM (`NumADPCMCoeff`, Annex C.3.3).
pub const ADPCM_ORDER: usize = 4;
/// Number of vectors in the D.10.2 high-frequency code book (10-bit index).
pub const HF_VQ_VECTORS: usize = 1024;
/// Elements per high-frequency vector (4 subsubframes × 8 samples).
pub const HF_VQ_LEN: usize = 32;

/// The D.10.1 ADPCM prediction-coefficient code book.
#[derive(Clone, PartialEq, Eq)]
pub struct AdpcmCodebook {
    /// `entries[pvq][n]`: coefficient n of vector `pvq`, × 2¹³.
    entries: Box<[[i16; ADPCM_ORDER]]>,
}

impl fmt::Debug for AdpcmCodebook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdpcmCodebook {{ first: {:?}, .. }}", self.entries[0])
    }
}

impl AdpcmCodebook {
    /// From the 4 096 printed entries, in index order, each as four
    /// coefficients × 2¹³.
    pub fn from_entries(entries: &[[i16; ADPCM_ORDER]]) -> Result<Self, Error> {
        if entries.len() != ADPCM_VECTORS {
            return Err(Error::Invalid("an ADPCM code book has exactly 4096 vectors"));
        }
        Ok(Self { entries: entries.into() })
    }

    /// From 32 768 bytes: 4 096 × 4 big-endian 16-bit two's-complement
    /// entries, vector after vector.
    pub fn from_be_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != ADPCM_VECTORS * ADPCM_ORDER * 2 {
            return Err(Error::Invalid("an ADPCM code book file is 32768 bytes (4096 × 4 × i16 BE)"));
        }
        let entries: Vec<[i16; ADPCM_ORDER]> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| std::array::from_fn(|n| i16::from_be_bytes([v[2 * n], v[2 * n + 1]])))
            .collect();
        Self::from_entries(&entries)
    }

    /// The entries as stored (× 2¹³).
    pub fn entries(&self) -> &[[i16; ADPCM_ORDER]] {
        &self.entries
    }

    /// Vector `pvq` as real coefficients (`Entry / 2¹³`, D.10.1):
    /// `x[m] = residual[m] + Σ c[n]·x[m−n−1]`.
    pub fn coefficients(&self, pvq: usize) -> [f64; ADPCM_ORDER] {
        let v = self.entries[pvq];
        std::array::from_fn(|n| v[n] as f64 / 8192.0)
    }

    /// A deterministic code book of 4 096 stable 4th-order predictors, built
    /// here from two resonator sections each (8 pole radii × 8 pole angles
    /// per section). It is **not** the D.10.1 code book: a stream predicted
    /// with it is valid DTS syntax but decodes correctly only on a decoder
    /// given this same book. It exists to exercise the prediction paths.
    pub fn private_test_book() -> Self {
        const RADII: [f64; 8] = [0.0, 0.5, 0.7, 0.8, 0.88, 0.93, 0.96, 0.98];
        let mut entries = Vec::with_capacity(ADPCM_VECTORS);
        for i in 0..ADPCM_VECTORS {
            let (s1, s2) = (i & 63, i >> 6);
            // Section s: radius RADII[s & 7], angle (s >> 3 + 0.5)·π/8. The
            // stored range is ±4, so a pair whose product overflows it has
            // both radii pulled in until it fits (still stable).
            let mut shrink = 1.0;
            loop {
                let section = |s: usize| {
                    let r = RADII[s & 7] * shrink;
                    let th = ((s >> 3) as f64 + 0.5) * std::f64::consts::PI / 8.0;
                    [1.0, -2.0 * r * th.cos(), r * r]
                };
                let (a, b) = (section(s1), section(s2));
                // A(z) = a(z)·b(z) = 1 + p1 z⁻¹ + … + p4 z⁻⁴; predictor c = −p.
                let p = [
                    a[1] + b[1],
                    a[2] + a[1] * b[1] + b[2],
                    a[2] * b[1] + a[1] * b[2],
                    a[2] * b[2],
                ];
                let q: [f64; ADPCM_ORDER] = std::array::from_fn(|n| (-p[n] * 8192.0).round());
                if q.iter().all(|v| v.abs() <= i16::MAX as f64) {
                    entries.push(q.map(|v| v as i16));
                    break;
                }
                shrink *= 0.97;
            }
        }
        Self { entries: entries.into() }
    }
}

/// The D.10.2 high-frequency subband VQ code book.
#[derive(Clone, PartialEq, Eq)]
pub struct HfVqCodebook {
    /// `entries[index][m]`: element m of vector `index`, × 2⁴.
    entries: Box<[[i8; HF_VQ_LEN]]>,
}

impl fmt::Debug for HfVqCodebook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HfVqCodebook {{ first: {:?}, .. }}", &self.entries[0][..4])
    }
}

impl HfVqCodebook {
    /// From the 1 024 vectors of 32 elements, each element × 2⁴.
    pub fn from_entries(entries: &[[i8; HF_VQ_LEN]]) -> Result<Self, Error> {
        if entries.len() != HF_VQ_VECTORS {
            return Err(Error::Invalid("a high-frequency VQ code book has exactly 1024 vectors"));
        }
        Ok(Self { entries: entries.into() })
    }

    /// From the table as printed: 16 384 16-bit entries (big-endian here),
    /// 16 per vector, each holding two elements, high byte first.
    pub fn from_be_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != HF_VQ_VECTORS * HF_VQ_LEN {
            return Err(Error::Invalid("a high-frequency VQ code book file is 32768 bytes"));
        }
        let entries: Vec<[i8; HF_VQ_LEN]> = bytes
            .as_chunks::<HF_VQ_LEN>()
            .0
            .iter()
            .map(|v| std::array::from_fn(|m| v[m] as i8))
            .collect();
        Self::from_entries(&entries)
    }

    /// The entries as stored (× 2⁴).
    pub fn entries(&self) -> &[[i8; HF_VQ_LEN]] {
        &self.entries
    }

    /// Element `m` of vector `index` as a real value (`int8 / 2⁴`).
    pub fn element(&self, index: usize, m: usize) -> f64 {
        self.entries[index][m] as f64 / 16.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every vector of the test book is a stable predictor: the roots of
    /// 1 − Σ c[n] z⁻ⁿ⁻¹ lie inside the unit circle. Checked through the
    /// impulse response of the all-pole filter decaying.
    #[test]
    fn test_book_predictors_are_stable() {
        let book = AdpcmCodebook::private_test_book();
        for pvq in 0..ADPCM_VECTORS {
            let c = book.coefficients(pvq);
            let mut h = [0.0f64; 4];
            let mut x = 1.0;
            let mut peak_late = 0.0f64;
            for m in 0..4000 {
                let y = x + c[0] * h[0] + c[1] * h[1] + c[2] * h[2] + c[3] * h[3];
                x = 0.0;
                h = [y, h[0], h[1], h[2]];
                if m > 3000 {
                    peak_late = peak_late.max(y.abs());
                }
            }
            assert!(peak_late < 1e-3, "PVQ {pvq} {c:?} rings at {peak_late}");
        }
    }

    #[test]
    fn byte_forms_round_trip() {
        let book = AdpcmCodebook::private_test_book();
        let bytes: Vec<u8> = book.entries().iter().flatten().flat_map(|v| v.to_be_bytes()).collect();
        assert_eq!(AdpcmCodebook::from_be_bytes(&bytes).unwrap(), book);
        assert!(AdpcmCodebook::from_be_bytes(&bytes[1..]).is_err());
        let hf: Vec<u8> = (0..HF_VQ_VECTORS * HF_VQ_LEN).map(|i| (i * 7) as u8).collect();
        let hfb = HfVqCodebook::from_be_bytes(&hf).unwrap();
        assert_eq!(hfb.element(0, 1), 7.0 / 16.0);
        assert_eq!(hfb.element(1, 4), ((32 + 4) * 7 % 256) as u8 as i8 as f64 / 16.0);
    }
}
