//! DTS Coherent Acoustics **core** encoder.
//!
//! Produces core frames (ETSI TS 102 114 clause 5) that this crate's
//! [`Decoder`](crate::Decoder) — and any core decoder — reads: 16-bit
//! big-endian framing, 512 samples per channel per frame (`NBLKS` = 15),
//! one subframe of two subsubframes, constant frame size for the chosen
//! bit rate (`CPF` = 0, so the core carries no CRC words; `DSYNC` closes
//! every subframe).
//!
//! The coding follows the encoder the specification outlines (§5.1,
//! Figure 5-1):
//!
//! - a 32-band analysis bank matched to the decoder's synthesis bank
//!   (`FILTS` = 1, the perfect-reconstruction prototype, by default);
//! - per subband, scale factors from the 6-bit square-root table (D.1.1)
//!   and a mid-tread quantiser chosen by bit allocation (`ABITS`), with a
//!   second scale factor after a transient (`TMODE`);
//! - bit allocation by greedy noise-to-mask ratio: each step gives one more
//!   `ABITS` to the subband whose quantisation noise is furthest above a
//!   simple masking threshold (spread subband energies and the absolute
//!   threshold of hearing), costing every step exactly;
//! - per channel and `ABITS`, the cheapest of the Huffman books, block code
//!   or plain two's complement of Table 5-26 (`SEL`), and the cheapest of
//!   the scale-factor codes (`SHUFF`);
//! - subband ADPCM (`PMODE`/`PVQ`, Annex C.3.3) where the prediction gain
//!   pays for the coefficients, when a D.10.1 code book is supplied;
//! - the LFE decimated 64× (`LFF` = 2) by the filter matched to the
//!   decoder's interpolator and quantised to 8 bits (step 0.035, Table 5-29);
//! - rate control by construction: the allocation spends exactly the bits
//!   the frame has, and the frame is zero-padded to its constant size.
//!
//! Not used: high-frequency VQ (the D.10.2 code book is not published),
//! joint intensity and sum/difference coding, Huffman-coded `ABITS` (the
//! printed D.5.6 books carry levels 1–12, which cannot express `ABITS` = 0;
//! the linear 4/5-bit forms are unambiguous).

mod adpcm;
mod analysis;
mod bitwriter;
mod entropy;

use std::sync::Arc;

use crate::tables;
use crate::vq::{ADPCM_ORDER, AdpcmCodebook};
use crate::{CORE_SYNC, Error, Layout};
use analysis::{AnalysisBank, BANDS, DELAY, IR_LEN, LFE_FACTOR, LFE_TAPS, LfeDecimator};
use bitwriter::BitWriter;
use entropy::Coding;

/// Samples per channel in one encoded frame.
pub const FRAME_SAMPLES: usize = 512;
/// Subband samples per subband per frame (`NBLKS` + 1).
const SUBBAND_SAMPLES: usize = FRAME_SAMPLES / 32;
/// Subsubframes in the frame's single subframe.
const SSC: usize = 2;
/// Decimated LFE samples per frame (`2·LFF·nSSC`, `LFF` = 2).
const LFE_SAMPLES: usize = FRAME_SAMPLES / LFE_FACTOR;
/// Input lookahead one frame needs past its first sample.
const LOOKAHEAD: usize = 32 * (SUBBAND_SAMPLES - 1) + IR_LEN;
/// Full scale of the core's PCM domain (24-bit, as the decoder's output scale).
const PCM_SCALE: f64 = (1u32 << 23) as f64;
/// Largest `ABITS` (Table 5-26).
const MAX_ABITS: usize = 26;

/// Table 5-7: `RATE` codes and their bit rates in bits per second.
const RATES: [(u32, u32); 25] = [
    (0, 32_000),
    (1, 56_000),
    (2, 64_000),
    (3, 96_000),
    (4, 112_000),
    (5, 128_000),
    (6, 192_000),
    (7, 224_000),
    (8, 256_000),
    (9, 320_000),
    (10, 384_000),
    (11, 448_000),
    (12, 512_000),
    (13, 576_000),
    (14, 640_000),
    (15, 768_000),
    (16, 960_000),
    (17, 1_024_000),
    (18, 1_152_000),
    (19, 1_280_000),
    (20, 1_344_000),
    (21, 1_408_000),
    (22, 1_411_200),
    (23, 1_472_000),
    (24, 1_536_000),
];

/// What to encode and how.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Core sample rate: 48 000, 44 100 or 32 000 Hz.
    pub sample_rate: u32,
    /// Channel layout; the input is interleaved in its speaker order (the
    /// order the decoder outputs). Any core arrangement: mono, stereo, 3.0,
    /// 3.0(back), 4.0, quad(side), 5.0(side), each with or without the LFE.
    pub layout: Layout,
    /// Target bit rate in bits per second: one of the Table 5-7 rates
    /// (32 000 … 1 536 000, including 1 411 200). The frame size is the
    /// largest even byte count at or under it.
    pub bit_rate: u32,
    /// The D.10.1 prediction code book. `None`: no subband is predicted
    /// (`PMODE` = 0 throughout). `Some`: ADPCM where it pays. Give the
    /// genuine D.10.1 book for streams other decoders can read;
    /// [`AdpcmCodebook::private_test_book`] only round-trips through this
    /// crate's decoder given the same book.
    pub adpcm_codebook: Option<Arc<AdpcmCodebook>>,
    /// `FILTS`: the perfect-reconstruction QMF prototype (default `true`)
    /// or the non-perfect one.
    pub perfect_reconstruction: bool,
    /// Detect transients and send a second scale factor (`TMODE`);
    /// default `true`.
    pub transients: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            layout: Layout::Surround51Side,
            bit_rate: 1_536_000,
            adpcm_codebook: None,
            perfect_reconstruction: true,
            transients: true,
        }
    }
}

impl EncoderConfig {
    /// A configuration with the defaults for the other fields.
    pub fn new(sample_rate: u32, layout: Layout, bit_rate: u32) -> Self {
        Self { sample_rate, layout, bit_rate, ..Self::default() }
    }
}

/// One subband's samples for this frame and its quantisation options.
struct Band {
    /// A second scale factor from the second subsubframe on.
    tmode: bool,
    /// ADPCM: `(PVQ, coefficients)`.
    pred: Option<(usize, [f64; ADPCM_ORDER])>,
    /// `opts[a]` for `ABITS` = a; `None` where infeasible.
    opts: Vec<Opt>,
    /// Masking threshold over the frame's 16 samples (energy).
    mask: f64,
    /// Current allocation.
    a: usize,
    /// Cannot be given more bits this frame.
    frozen: bool,
}

/// One `ABITS` choice for a band.
struct Opt {
    q: [i32; SUBBAND_SAMPLES],
    /// 6-bit scale indices: [pre-transient, post-transient].
    sf: [usize; 2],
    /// Reconstruction the decoder will produce.
    rec: [f64; SUBBAND_SAMPLES],
    noise: f64,
    /// Sample bits per `SEL` (`u32::MAX`: not codable).
    cost: Vec<u32>,
}

/// A DTS core encoder. One per stream: the filter bank and the ADPCM
/// history carry from frame to frame.
pub struct Encoder {
    cfg: EncoderConfig,
    amode: u32,
    sfreq: u32,
    rate: u32,
    frame_bytes: usize,
    /// Core channel `c` reads input slot `core_slots[c]`.
    core_slots: Vec<usize>,
    lfe_slot: Option<usize>,
    in_channels: usize,
    bank: &'static AnalysisBank,
    /// Per core channel, the (delayed) input not yet consumed.
    buf: Vec<Vec<f64>>,
    lfe_buf: Vec<f64>,
    /// Real input samples per channel received so far.
    received: u64,
    frames: u64,
    /// Per core channel and subband, the last four reconstructed subband
    /// samples, newest first (the decoder's ADPCM history, `HFLAG` = 1).
    hist: Vec<[[f64; ADPCM_ORDER]; BANDS]>,
    /// Masking threshold floor per subband (absolute threshold), subband
    /// domain energy per sample.
    ath: [f64; BANDS],
}

impl Encoder {
    /// Validate `cfg` and set up the encoder.
    pub fn new(cfg: EncoderConfig) -> Result<Self, Error> {
        let sfreq = match cfg.sample_rate {
            48_000 => 13,
            44_100 => 8,
            32_000 => 3,
            r => return Err(Error::Unsupported(format!("encoder sample rate {r} Hz (48000, 44100 or 32000)"))),
        };
        let rate = RATES
            .iter()
            .find(|(_, r)| *r == cfg.bit_rate)
            .map(|(c, _)| *c)
            .ok_or_else(|| Error::Unsupported(format!("bit rate {} b/s is not a Table 5-7 rate", cfg.bit_rate)))?;
        let frame_bytes =
            ((cfg.bit_rate as u64 * FRAME_SAMPLES as u64 / cfg.sample_rate as u64 / 8) & !1) as usize;
        if !(96..=16_384).contains(&frame_bytes) {
            return Err(Error::Unsupported(format!(
                "{} b/s at {} Hz gives {frame_bytes}-byte frames; FSIZE must be 95..=16383",
                cfg.bit_rate, cfg.sample_rate
            )));
        }
        // AMODE and the core channel order (C L R SL SR …) against the
        // layout's speaker order, as the decoder maps them: any core
        // arrangement (AMODE 0, 2, 5–9), with or without the LFE.
        let spk = cfg.layout.speakers();
        let lfe_slot = spk.iter().position(|s| *s == crate::Speaker::LFE);
        let (amode, core_slots) = [0u32, 2, 5, 6, 7, 8, 9]
            .into_iter()
            .find_map(|amode| {
                let core = crate::layout::amode_speakers(amode)?;
                let slots: Option<Vec<usize>> = core.iter().map(|c| spk.iter().position(|s| s == c)).collect();
                let slots = slots?;
                (slots.len() + lfe_slot.is_some() as usize == spk.len()).then_some((amode, slots))
            })
            .ok_or_else(|| {
                Error::Unsupported(format!("layout {} is not a core channel arrangement", cfg.layout))
            })?;
        let channels = core_slots.len();
        // Minimum frame: headers, side information for every band and a
        // little audio. Refuse configurations that cannot hold that.
        let min_bits = Self::fixed_bits(channels, lfe_slot.is_some()) + 64;
        if frame_bytes * 8 < min_bits {
            return Err(Error::Unsupported(format!(
                "{} b/s is too low for {} channels ({frame_bytes}-byte frames)",
                cfg.bit_rate,
                cfg.layout.channels()
            )));
        }
        let bank = AnalysisBank::get(cfg.perfect_reconstruction);
        let ath = Self::absolute_threshold(cfg.sample_rate, bank.synthesis_energy);
        Ok(Self {
            amode,
            sfreq,
            rate,
            frame_bytes,
            in_channels: cfg.layout.channels(),
            core_slots,
            lfe_slot,
            bank,
            buf: vec![vec![0.0; DELAY]; channels],
            lfe_buf: vec![0.0; DELAY],
            received: 0,
            frames: 0,
            hist: vec![[[0.0; ADPCM_ORDER]; BANDS]; channels],
            ath,
            cfg,
        })
    }

    /// Bytes per frame (`FSIZE` + 1); every frame has this size.
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// End-to-end delay: decoded sample `n` reconstructs input sample
    /// `n − delay()` (the analysis bank's lookahead, a whole number of
    /// blocks; the LFE path is aligned to it).
    pub fn delay(&self) -> usize {
        DELAY
    }

    /// The configuration in use.
    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// Feed interleaved samples (±1.0 full scale, `layout.channels()` per
    /// sample frame, in the layout's speaker order); returns every frame
    /// completed by them.
    pub fn encode(&mut self, interleaved: &[f32]) -> Result<Vec<Vec<u8>>, Error> {
        if !interleaved.len().is_multiple_of(self.in_channels) {
            return Err(Error::Invalid("interleaved input is not a whole number of sample frames"));
        }
        for frame in interleaved.chunks_exact(self.in_channels) {
            for (c, &slot) in self.core_slots.iter().enumerate() {
                self.buf[c].push(frame[slot] as f64 * PCM_SCALE);
            }
            if let Some(slot) = self.lfe_slot {
                self.lfe_buf.push(frame[slot] as f64 * PCM_SCALE);
            }
        }
        self.received += (interleaved.len() / self.in_channels) as u64;
        let mut out = Vec::new();
        while self.buf[0].len() >= LOOKAHEAD {
            out.push(self.encode_frame());
        }
        Ok(out)
    }

    /// Encode what is left, padded with silence, so that the decoded
    /// stream covers every input sample (after [`delay`](Self::delay)).
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>, Error> {
        let needed = (DELAY as u64 + self.received).div_ceil(FRAME_SAMPLES as u64);
        let mut out = Vec::new();
        while self.frames < needed {
            for b in self.buf.iter_mut().chain(std::iter::once(&mut self.lfe_buf)) {
                if b.len() < LOOKAHEAD {
                    b.resize(LOOKAHEAD, 0.0);
                }
            }
            out.push(self.encode_frame());
        }
        Ok(out)
    }

    /// Bits that do not depend on the allocation (upper bounds where the
    /// final choice can only be cheaper): headers, per-band `PMODE` and
    /// 5-bit `ABITS`, the LFE, `DSYNC`.
    fn fixed_bits(channels: usize, lfe: bool) -> usize {
        let bitstream_header = 88 + 16;
        let audio_header = 7 + channels * (5 + 5 + 3 + 2 + 3 + 3 + 1 + 4 * 2 + 5 * 3);
        let side = 5 + channels * BANDS * (1 + 5);
        let lfe_bits = if lfe { 8 * LFE_SAMPLES + 8 } else { 0 };
        bitstream_header + audio_header + side + lfe_bits + 16
    }

    /// The absolute threshold of hearing per subband (Terhardt's
    /// approximation, full scale taken as 96 dB SPL, capped at 60 dB SPL so
    /// the top subbands still get bits when the rate allows), converted to
    /// subband-domain noise energy per sample.
    fn absolute_threshold(sample_rate: u32, synthesis_energy: f64) -> [f64; BANDS] {
        std::array::from_fn(|sb| {
            let width = sample_rate as f64 / 64.0;
            let db = (0..8)
                .map(|i| {
                    let f = ((sb as f64 + (i as f64 + 0.5) / 8.0) * width).max(20.0) / 1000.0;
                    3.64 * f.powf(-0.8) - 6.5 * (-0.6 * (f - 3.3).powi(2)).exp() + 1e-3 * f.powi(4)
                })
                .fold(f64::INFINITY, f64::min)
                .min(60.0);
            let time_power = PCM_SCALE * PCM_SCALE / 2.0 * 10f64.powf((db - 96.0) / 10.0);
            time_power * 32.0 / synthesis_energy
        })
    }

    fn encode_frame(&mut self) -> Vec<u8> {
        let channels = self.core_slots.len();
        let lfe = self.lfe_slot.is_some();

        // Analysis.
        let mut bands: Vec<Vec<Band>> = Vec::with_capacity(channels);
        for c in 0..channels {
            let mut x = [[0.0f64; SUBBAND_SAMPLES]; BANDS];
            let mut blk = [0.0f64; BANDS];
            for s in 0..SUBBAND_SAMPLES {
                self.bank.analyse(&self.buf[c][32 * s..], &mut blk);
                for k in 0..BANDS {
                    x[k][s] = blk[k];
                }
            }
            bands.push(self.prepare_channel(c, &x));
        }
        let lfe_q = lfe.then(|| {
            let dec = LfeDecimator::get(self.cfg.sample_rate);
            let d: [f64; LFE_SAMPLES] = std::array::from_fn(|j| dec.decimate(&self.lfe_buf[64 * j..64 * j + LFE_TAPS]));
            quantise_lfe(&d)
        });

        // Allocation.
        let frame_bits = self.frame_bytes * 8;
        let budget = frame_bits.saturating_sub(Self::fixed_bits(channels, lfe));
        allocate(&mut bands, budget);

        // Packing.
        let bytes = self.pack(&bands, lfe_q.as_ref());
        debug_assert!(bytes.len() <= self.frame_bytes);

        // History for the next frame's prediction.
        for (c, chb) in bands.iter().enumerate() {
            for (k, b) in chb.iter().enumerate() {
                let rec = if b.a > 0 { b.opts[b.a].rec } else { [0.0; SUBBAND_SAMPLES] };
                self.hist[c][k] = std::array::from_fn(|n| rec[SUBBAND_SAMPLES - 1 - n]);
            }
        }
        for b in self.buf.iter_mut().chain(std::iter::once(&mut self.lfe_buf)) {
            b.drain(..FRAME_SAMPLES.min(b.len()));
        }
        self.frames += 1;
        bytes
    }

    /// Masking thresholds, transient decisions, prediction and the
    /// quantisation options for one channel's 32 subbands.
    fn prepare_channel(&self, c: usize, x: &[[f64; SUBBAND_SAMPLES]; BANDS]) -> Vec<Band> {
        let energy: [f64; BANDS] = std::array::from_fn(|k| x[k].iter().map(|v| v * v).sum::<f64>());
        // Spread the energies (≈ 16 dB below the masker, falling 10 dB per
        // band upwards and 20 dB per band downwards) and floor at the ATH.
        let mask: [f64; BANDS] = std::array::from_fn(|k| {
            let spread: f64 = (0..BANDS)
                .map(|j| {
                    let d = k as f64 - j as f64;
                    let att = 16.0 + if d >= 0.0 { 10.0 * d } else { -20.0 * d };
                    energy[j] * 10f64.powf(-att / 10.0)
                })
                .sum();
            spread.max(self.ath[k] * SUBBAND_SAMPLES as f64)
        });
        (0..BANDS)
            .map(|k| {
                let xk = x[k];
                let half = SUBBAND_SAMPLES / 2;
                let peak = |s: &[f64]| s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let (p0, p1) = (peak(&xk[..half]), peak(&xk[half..]));
                let tmode = self.cfg.transients && (p1 > 4.0 * p0 || p0 > 8.0 * p1) && p0.max(p1) > 64.0;
                let pred = self.cfg.adpcm_codebook.as_deref().and_then(|book| {
                    adpcm::best_vector(book, &self.hist[c][k], &xk).map(|(pvq, _)| (pvq, book.coefficients(pvq)))
                });
                let opts = (0..=MAX_ABITS).map(|a| make_opt(&xk, a, tmode, pred.map(|p| p.1), &self.hist[c][k])).collect();
                Band { tmode, pred, opts, mask: mask[k], a: 0, frozen: false }
            })
            .collect()
    }

    fn pack(&self, bands: &[Vec<Band>], lfe_q: Option<&([i32; LFE_SAMPLES], usize)>) -> Vec<u8> {
        let channels = bands.len();
        let mut w = BitWriter::new();
        // Bit stream header (Table 5-1).
        w.put(CORE_SYNC, 32);
        w.put(1, 1); // FTYPE: normal frame
        w.put(31, 5); // SHORT
        w.put(0, 1); // CPF
        w.put((SUBBAND_SAMPLES - 1) as u32, 7); // NBLKS
        w.put((self.frame_bytes - 1) as u32, 14); // FSIZE
        w.put(self.amode, 6);
        w.put(self.sfreq, 4);
        w.put(self.rate, 5);
        w.put(0, 1); // FixedBit
        w.put(0, 4); // DYNF TIMEF AUXF HDCD
        w.put(0, 3); // EXT_AUDIO_ID
        w.put(0, 1); // EXT_AUDIO
        w.put(0, 1); // ASPF: DSYNC per subframe
        w.put(if lfe_q.is_some() { 2 } else { 0 }, 2); // LFF: 64× interpolation
        w.put(1, 1); // HFLAG: predictor history carries across frames
        w.put(u32::from(self.cfg.perfect_reconstruction), 1); // FILTS
        w.put(7, 4); // VERNUM
        w.put(0, 2); // CHIST
        w.put(0b110, 3); // PCMR: 24-bit source
        w.put(0, 2); // SUMF SUMS
        w.put(0, 4); // DIALNORM: 0 dB

        // Per channel decisions.
        let nsubs: Vec<usize> = bands
            .iter()
            .map(|chb| chb.iter().rposition(|b| b.a > 0).map_or(2, |k| (k + 1).max(2)))
            .collect();
        let sel: Vec<[usize; 10]> = bands.iter().map(|chb| choose_sel(chb)).collect();
        let bhuff: Vec<u32> =
            bands.iter().map(|chb| if chb.iter().all(|b| b.a <= 15) { 5 } else { 6 }).collect();
        let scale_lists: Vec<Vec<usize>> = bands
            .iter()
            .zip(&nsubs)
            .map(|(chb, &n)| {
                let mut v = Vec::new();
                for b in &chb[..n] {
                    if b.a > 0 {
                        let o = &b.opts[b.a];
                        v.push(o.sf[0]);
                        if b.tmode {
                            v.push(o.sf[1]);
                        }
                    }
                }
                v
            })
            .collect();
        let shuff: Vec<u32> = scale_lists.iter().map(|l| choose_shuff(l)).collect();

        // Primary audio coding header (Table 5-21).
        w.put(0, 4); // SUBFS: one subframe
        w.put(channels as u32 - 1, 3); // PCHS
        for &n in &nsubs {
            w.put(n as u32 - 2, 5); // SUBS
        }
        for &n in &nsubs {
            w.put(n as u32 - 1, 5); // VQSUB: no VQ subbands
        }
        for _ in 0..channels {
            w.put(0, 3); // JOINX
        }
        for _ in 0..channels {
            w.put(0, 2); // THUFF: A4
        }
        for &s in &shuff {
            w.put(s, 3);
        }
        for &b in &bhuff {
            w.put(b, 3);
        }
        for s in &sel {
            w.put(s[0] as u32, 1);
        }
        for n in 1..5 {
            for s in &sel {
                w.put(s[n] as u32, 2);
            }
        }
        for n in 5..10 {
            for s in &sel {
                w.put(s[n] as u32, 3);
            }
        }
        // ADJ wherever SEL picked a Huffman book: index 0 (1.0).
        for n in 0..10 {
            for s in &sel {
                if entropy::codings(n as u32 + 1)[s[n]].is_huffman() {
                    w.put(0, 2);
                }
            }
        }

        // Subframe side information (Table 5-28).
        w.put(SSC as u32 - 1, 2); // SSC
        w.put(0, 3); // PSC
        let predicted = |b: &Band| b.pred.is_some() && b.a > 0;
        for (chb, &n) in bands.iter().zip(&nsubs) {
            for b in &chb[..n] {
                w.flag(predicted(b));
            }
        }
        for (chb, &n) in bands.iter().zip(&nsubs) {
            for b in &chb[..n] {
                if predicted(b) {
                    w.put(b.pred.expect("predicted").0 as u32, 12);
                }
            }
        }
        for ((chb, &n), &bh) in bands.iter().zip(&nsubs).zip(&bhuff) {
            for b in &chb[..n] {
                w.put(b.a as u32, if bh == 5 { 4 } else { 5 });
            }
        }
        for (chb, &n) in bands.iter().zip(&nsubs) {
            for b in &chb[..n] {
                if b.a > 0 {
                    entropy::tmode_book().put(&mut w, i32::from(b.tmode));
                }
            }
        }
        for (list, &s) in scale_lists.iter().zip(&shuff) {
            let mut prev = 0i32;
            for &idx in list {
                if s == 5 {
                    w.put(idx as u32, 6);
                } else {
                    entropy::scale_book(s).put(&mut w, idx as i32 - prev);
                }
                prev = idx as i32;
            }
        }
        // (No joint intensity, no DYNF, CPF = 0.)

        // LFE (Table 5-29).
        if let Some((q, idx)) = lfe_q {
            for &v in q {
                w.put_signed(v, 8);
            }
            w.put(*idx as u32, 8);
        }

        // Audio data arrays.
        for ssf in 0..SSC {
            for ((chb, &n), s) in bands.iter().zip(&nsubs).zip(&sel) {
                for b in &chb[..n] {
                    if b.a == 0 {
                        continue;
                    }
                    let coding = coding_for(b.a, s);
                    coding.put(&mut w, &b.opts[b.a].q[ssf * 8..ssf * 8 + 8]);
                }
            }
        }
        w.put(0xFFFF, 16); // DSYNC

        let used = w.len_bits();
        assert!(
            used <= self.frame_bytes * 8,
            "rate control overran the frame: {used} bits for {} bytes",
            self.frame_bytes
        );
        let mut bytes = w.into_bytes();
        bytes.resize(self.frame_bytes, 0);
        bytes
    }
}

/// The coding a band with `ABITS` = `a` is written with, given the
/// channel's `SEL` choices.
fn coding_for(a: usize, sel: &[usize; 10]) -> Coding {
    if a <= 10 { entropy::codings(a as u32)[sel[a - 1]] } else { entropy::codings(a as u32)[0] }
}

/// The quantisation of one band at `ABITS` = `a`.
fn make_opt(x: &[f64; SUBBAND_SAMPLES], a: usize, tmode: bool, pred: Option<[f64; ADPCM_ORDER]>, hist: &[f64; ADPCM_ORDER]) -> Opt {
    let mut opt = Opt {
        q: [0; SUBBAND_SAMPLES],
        sf: [0; 2],
        rec: [0.0; SUBBAND_SAMPLES],
        noise: 0.0,
        cost: Vec::new(),
    };
    if a == 0 {
        opt.noise = x.iter().map(|v| v * v).sum();
        return opt;
    }
    let step = tables::STEP_SIZE_LOSSY_Q22[a] as f64 / (1u32 << 22) as f64;
    let qmax = entropy::max_index(a as u32);
    let half = SUBBAND_SAMPLES / 2;
    // What is quantised: the samples, or (open loop, for the scale factor)
    // the prediction residual.
    let target: [f64; SUBBAND_SAMPLES] = match pred {
        None => *x,
        Some(c) => {
            let mut h = *hist;
            std::array::from_fn(|m| {
                let p: f64 = (0..ADPCM_ORDER).map(|n| c[n] * h[n]).sum();
                h.rotate_right(1);
                h[0] = x[m];
                x[m] - p
            })
        }
    };
    let seg_peak = |r: std::ops::Range<usize>| target[r].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let pick_sf = |peak: f64| {
        let reach = qmax as f64 * step;
        (0..63).find(|&i| tables::SCALE_RMS_6BIT[i] as f64 * reach >= peak).unwrap_or(62)
    };
    if tmode {
        opt.sf = [pick_sf(seg_peak(0..half)), pick_sf(seg_peak(half..SUBBAND_SAMPLES))];
    } else {
        let s = pick_sf(seg_peak(0..SUBBAND_SAMPLES));
        opt.sf = [s, s];
    }
    let mut h = *hist;
    for m in 0..SUBBAND_SAMPLES {
        let delta = step * tables::SCALE_RMS_6BIT[opt.sf[usize::from(m >= half)]] as f64;
        let p = pred.map_or(0.0, |c| (0..ADPCM_ORDER).map(|n| c[n] * h[n]).sum());
        let q = ((x[m] - p) / delta).round().clamp(-qmax as f64, qmax as f64) as i32;
        let rec = q as f64 * delta + p;
        opt.q[m] = q;
        opt.rec[m] = rec;
        opt.noise += (x[m] - rec).powi(2);
        h.rotate_right(1);
        h[0] = rec;
    }
    opt.cost = entropy::codings(a as u32)
        .iter()
        .map(|c| {
            let a = c.cost(&opt.q[..8]);
            let b = c.cost(&opt.q[8..]);
            match (a, b) {
                (Some(a), Some(b)) => a + b,
                _ => u32::MAX,
            }
        })
        .collect();
    opt
}

/// Side bits of a band at `ABITS` > 0 beyond the fixed `PMODE`/`ABITS`
/// fields (upper bounds: linear 6-bit scales).
fn side_bits(b: &Band) -> usize {
    let scales = if b.tmode { 12 } else { 6 };
    let tmode = if b.tmode { 2 } else { 1 };
    let pvq = if b.pred.is_some() { 12 } else { 0 };
    scales + tmode + pvq
}

/// Exact sample bits of one (channel, `ABITS`) group under its best `SEL`
/// (plus the 2-bit `ADJ` a Huffman choice costs).
fn group_bits(chb: &[Band], a: usize) -> u64 {
    if a == 0 {
        return 0;
    }
    let n = entropy::codings(a as u32).len();
    (0..n)
        .map(|sel| {
            let mut sum = 0u64;
            for b in chb.iter().filter(|b| b.a == a) {
                let c = b.opts[a].cost[sel];
                if c == u32::MAX {
                    return u64::MAX;
                }
                sum += c as u64;
            }
            let adj = if a <= 10 && entropy::codings(a as u32)[sel].is_huffman() && sum > 0 { 2 } else { 0 };
            sum + adj
        })
        .min()
        .unwrap_or(u64::MAX)
}

/// `SEL` per `ABITS` 1..=10 for one channel: the cheapest coding of the
/// subbands allocated that `ABITS`; the non-Huffman one (no `ADJ`) where
/// none is.
fn choose_sel(chb: &[Band]) -> [usize; 10] {
    std::array::from_fn(|i| {
        let a = i + 1;
        let codings = entropy::codings(a as u32);
        let mut best = (codings.len() - 1, u64::MAX);
        for sel in 0..codings.len() {
            let mut sum = 0u64;
            let mut any = false;
            for b in chb.iter().filter(|b| b.a == a) {
                any = true;
                let c = b.opts[a].cost[sel];
                sum = if c == u32::MAX { u64::MAX } else { sum.saturating_add(c as u64) };
            }
            if !any {
                return codings.len() - 1;
            }
            let total = sum.saturating_add(if codings[sel].is_huffman() { 2 } else { 0 });
            if total < best.1 {
                best = (sel, total);
            }
        }
        best.0
    })
}

/// `SHUFF` for a channel's scale indices: the cheapest of the five
/// difference books and the linear 6-bit form (5).
fn choose_shuff(list: &[usize]) -> u32 {
    let mut best = (5u32, 6 * list.len() as u64);
    for s in 0..5 {
        let book = entropy::scale_book(s);
        let mut prev = 0i32;
        let mut bits = 0u64;
        for &idx in list {
            match book.len(idx as i32 - prev) {
                Some(l) => bits += l as u64,
                None => {
                    bits = u64::MAX;
                    break;
                }
            }
            prev = idx as i32;
        }
        if bits < best.1 {
            best = (s, bits);
        }
    }
    best.0
}

/// Greedy noise-to-mask allocation within `budget` bits (the bits left
/// after [`Encoder::fixed_bits`]).
fn allocate(bands: &mut [Vec<Band>], budget: usize) {
    let mut groups: Vec<[u64; MAX_ABITS + 1]> = vec![[0; MAX_ABITS + 1]; bands.len()];
    let mut used: u64 = 0;
    loop {
        // The band whose noise is furthest above its mask and can grow.
        let mut best: Option<(usize, usize, f64)> = None;
        for (c, chb) in bands.iter().enumerate() {
            for (k, b) in chb.iter().enumerate() {
                if b.frozen || b.a >= MAX_ABITS {
                    continue;
                }
                let noise = b.opts[b.a].noise;
                if noise <= 0.0 {
                    continue;
                }
                let nmr = noise / b.mask.max(1e-30);
                if best.is_none_or(|(_, _, v)| nmr > v) {
                    best = Some((c, k, nmr));
                }
            }
        }
        let Some((c, k, _)) = best else { break };
        let from = bands[c][k].a;
        // Next ABITS that is codable and lowers the noise.
        let to = (from + 1..=MAX_ABITS).find(|&a| {
            let o = &bands[c][k].opts[a];
            o.cost.iter().any(|&v| v != u32::MAX) && o.noise < bands[c][k].opts[from].noise
        });
        let Some(to) = to else {
            bands[c][k].frozen = true;
            continue;
        };
        bands[c][k].a = to;
        let new_from = group_bits(&bands[c], from);
        let new_to = group_bits(&bands[c], to);
        let side = if from == 0 { side_bits(&bands[c][k]) as u64 } else { 0 };
        let next = used
            .checked_sub(groups[c][from] + groups[c][to])
            .and_then(|u| u.checked_add(new_from.checked_add(new_to)?))
            .and_then(|u| u.checked_add(side));
        match next {
            Some(n) if n <= budget as u64 => {
                used = n;
                groups[c][from] = new_from;
                groups[c][to] = new_to;
            }
            _ => {
                bands[c][k].a = from;
                bands[c][k].frozen = true;
            }
        }
    }
}

/// LFE: 8-bit indices and the 7-bit-table scale index (Table 5-29: the
/// decoder multiplies by `SCALE·0.035`).
fn quantise_lfe(d: &[f64; LFE_SAMPLES]) -> ([i32; LFE_SAMPLES], usize) {
    let peak = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let idx = (0..125)
        .find(|&i| tables::SCALE_RMS_7BIT[i] as f64 * 0.035 * 127.0 >= peak)
        .unwrap_or(124);
    let step = tables::SCALE_RMS_7BIT[idx] as f64 * 0.035;
    (d.map(|v| (v / step).round().clamp(-128.0, 127.0) as i32), idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_configurations() {
        assert!(Encoder::new(EncoderConfig::new(96_000, Layout::Stereo, 768_000)).is_err());
        assert!(Encoder::new(EncoderConfig::new(48_000, Layout::Stereo, 700_000)).is_err());
        assert!(Encoder::new(EncoderConfig::new(48_000, Layout::Surround51Side, 32_000)).is_err());
        let e = Encoder::new(EncoderConfig::new(48_000, Layout::Stereo, 1_536_000)).unwrap();
        assert_eq!(e.frame_bytes(), 2048);
        let e = Encoder::new(EncoderConfig::new(44_100, Layout::Stereo, 1_411_200)).unwrap();
        assert_eq!(e.frame_bytes(), 2048);
    }

    /// The closed-loop ADPCM quantisation reconstructs what Annex C.3.3
    /// (written out as a float reference) makes of the residuals.
    #[test]
    fn adpcm_quantisation_matches_the_reference_inverse() {
        let book = AdpcmCodebook::private_test_book();
        let sig: Vec<f64> = (0..36).map(|n| 3.0e5 * (0.7 * n as f64).sin() + 2.0e4 * (2.1 * n as f64).cos()).collect();
        let hist = [sig[3], sig[2], sig[1], sig[0]];
        let x: [f64; 16] = std::array::from_fn(|m| sig[4 + m]);
        let (pvq, gain) = adpcm::best_vector(&book, &hist, &x).expect("predictable");
        let c = book.coefficients(pvq);
        for a in [3usize, 6, 9, 12] {
            let o = make_opt(&x, a, false, Some(c), &hist);
            let step = tables::STEP_SIZE_LOSSY_Q22[a] as f64 / (1u32 << 22) as f64;
            let delta = step * tables::SCALE_RMS_6BIT[o.sf[0]] as f64;
            let residual: Vec<f64> = o.q.iter().map(|q| *q as f64 * delta).collect();
            let mut h = hist;
            let rec = adpcm::inverse_adpcm(&c, &mut h, &residual);
            for m in 0..16 {
                assert!((rec[m] - o.rec[m]).abs() < 1e-6, "ABITS {a} m {m}");
            }
            let plain = make_opt(&x, a, false, None, &hist);
            let snr = |n: f64| 10.0 * (x.iter().map(|v| v * v).sum::<f64>() / n).log10();
            eprintln!(
                "ABITS {a}: predicted (gain {:.1} dB) SNR {:.1} dB vs unpredicted {:.1} dB",
                10.0 * gain.log10(),
                snr(o.noise),
                snr(plain.noise)
            );
            assert!(o.noise < plain.noise, "prediction must help a predictable signal at ABITS {a}");
        }
    }

    #[test]
    fn lfe_quantiser_covers_its_peak() {
        let d = [1.0e6, -2.0e6, 0.0, 5.0, 3.0e6, -3.0e6, 1.0, 0.0];
        let (q, idx) = quantise_lfe(&d);
        let step = tables::SCALE_RMS_7BIT[idx] as f64 * 0.035;
        for (v, q) in d.iter().zip(q) {
            assert!((v - q as f64 * step).abs() <= step / 2.0 + 1e-9);
        }
    }
}
