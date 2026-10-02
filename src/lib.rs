//! DTS Coherent Acoustics decoder and core encoder, written from ETSI TS
//! 102 114 V1.6.1.
//!
//! The [`Decoder`] takes DTS packets — core frames (16-bit big-endian
//! framing), each optionally followed by DTS-HD extension substream frames
//! — and returns interleaved `f32` PCM. It decodes:
//!
//! - the **core** (clause 5): up to five full-band channels plus LFE at 8 to
//!   48 kHz, with every coding tool of the core: Huffman, block and linear
//!   quantisation indices, transients, joint intensity, sum/difference,
//!   dynamic range control, both QMF prototypes, 64×/128× LFE interpolation,
//!   **ADPCM prediction** and **high-frequency VQ** — the latter two with
//!   the caveat below;
//! - the **XCh** extension (6.4): the 6.1 back centre, undoing its embedded
//!   −3 dB downmix into the surrounds;
//! - the **XXCH** extension (6.5), in the core or in the extension
//!   substream: up to 7.1 and beyond, by loudspeaker mask, undoing embedded
//!   downmixes;
//! - the **X96** extension (6.2), in the core or in the extension substream:
//!   64-band synthesis at twice the core rate (88.2/96 kHz);
//! - the **XBR** extension (6.3): residuals that raise the core's resolution;
//! - the **DTS-HD extension substream** (clause 7): header and asset
//!   descriptors, so a DTS-HD High Resolution or Master Audio stream decodes
//!   to its core plus whichever of the above it carries, and a core carried
//!   inside the substream (sync 0x02B09261) decodes too.
//!
//! Not decoded: the lossless XLL extension (Master Audio's lossless layer;
//! its presence is reported in [`CoreInfo::skipped`]) and LBR (DTS Express).
//!
//! **The two code books ETSI does not print.** ADPCM prediction needs the
//! D.10.1 coefficient code book and VQ-coded subbands the D.10.2 one; no
//! edition of the specification includes them ("Due to its extensive size,
//! this table is not included here"), and this crate takes nothing from
//! other implementations. So, by default, a frame that predicts is refused
//! by name ([`Error::Unsupported`]) and VQ subbands decode as silence (which
//! §5.4.3 allows). A caller with a lawful copy supplies them
//! ([`Decoder::set_adpcm_codebook`], [`Decoder::set_hf_vq_codebook`]) and
//! decoding is then complete; or opts into [`AdpcmFallback::Estimate`],
//! which decodes predicted subbands with a predictor estimated from their
//! own history — an approximation, counted by [`Decoder::adpcm_estimated`].
//!
//! ```no_run
//! # fn main() -> Result<(), dts::Error> {
//! # let packets: Vec<Vec<u8>> = Vec::new();
//! let mut dec = dts::Decoder::new();
//! for packet in &packets {
//!     for frame in dec.decode(packet)? {
//!         // frame.samples: interleaved f32, ±1.0 full scale;
//!         // frame.sample_rate, frame.channels, frame.layout
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! The [`Encoder`] writes DTS core frames (mono to 5.1, 32/44.1/48 kHz, the
//! standard bit rates) that any core decoder reads.
//!
//! Output channels are always in the canonical [`Speaker`] order, named by
//! [`Layout`]. Not applied: dialog normalisation and embedded down-mix
//! coefficients for down-mixing (the decoder outputs the full layout).

// The unpacking loops index several per-channel/per-subband arrays by the
// same `ch` / `sb` the spec's pseudocode uses, so they read against it
// line by line; iterator forms would not.
#![allow(clippy::needless_range_loop)]

mod adpcm;
mod bits;
mod core;
mod crc;
pub mod encoder;
mod exss;
mod ext;
mod huffman;
mod layout;
mod synth;
pub mod tables;
#[cfg(test)]
mod tests;
pub mod vq;

use std::sync::Arc;

use bits::BitReader;
use core::{AudioCtx, ChannelBuf, CodingParams, NSB};
use ext::CoreFrameCtx;
use synth::{Lfe2x, LfeInterp, Qmf, Qmf64};

pub use adpcm::AdpcmFallback;
pub use encoder::{Encoder, EncoderConfig};
pub use layout::{Layout, Speaker, SpeakerList};
pub use vq::{AdpcmCodebook, HfVqCodebook};

/// Core substream sync word (§5.3), 16-bit big-endian framing — the only
/// framing that occurs inside Matroska/MP4. See [`normalize_framing`] for
/// raw streams in the other three.
pub const CORE_SYNC: u32 = 0x7FFE_8001;
/// DTS-HD extension substream sync word (Table 7-1).
pub const EXSS_SYNC: u32 = exss::SYNC_EXSS;

/// Why a packet did not decode. The messages name the field or the missing
/// table; [`Error::Unsupported`] is valid DTS this decoder does not decode,
/// the rest are broken input.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The frame ends before its fields do: `at_bit` is where the read
    /// started, `wanted` how many bits it needed.
    #[error("DTS: frame truncated at bit {at_bit} (wanted {wanted} more bits)")]
    Truncated { at_bit: usize, wanted: u32 },
    /// The packet does not start with [`CORE_SYNC`] (or [`EXSS_SYNC`]).
    #[error("DTS: packet does not start with the core sync word 0x7FFE8001")]
    NoSync,
    /// A field out of range or a structure that does not add up.
    #[error("DTS: invalid bitstream: {0}")]
    Invalid(&'static str),
    /// Valid DTS this decoder does not decode, named: ADPCM prediction
    /// without the code book, termination frames, channel arrangements with
    /// no defined speakers, an incompatible encoder revision, an extension
    /// substream with no core.
    #[error("DTS: unsupported: {0}")]
    Unsupported(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Core sample rates by `SFREQ` (Table 5-5); 0 marks a reserved code.
const SAMPLE_RATES: [u32; 16] = [
    0, 8_000, 16_000, 32_000, 0, 0, 11_025, 22_050, 44_100, 0, 0, 12_000, 24_000, 48_000, 0, 0,
];

/// Full-range channel count per `AMODE` (Table 5-4).
const AMODE_CHANNELS: [usize; 16] = [1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6, 7, 8, 8];

/// The subband-sample / scale-factor domain is 24-bit PCM: the scale factor
/// tables (D.1) top out at 8 317 638 ≈ 2^23 (138.4 dB), and Annex C writes
/// the QMF and LFE outputs as `int` samples on that same scale.
const OUTPUT_SCALE: f64 = 1.0 / (1u32 << 23) as f64;

/// Maximum primary channels the core carries (§5.4.3: `nPCHS`).
const MAX_CHANNELS: usize = 5;

/// −3 dB, the XCh downmix of the back centre into each surround (6.1).
const XCH_DOWNMIX: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A left/right pair of core channel indices.
type Pair = (usize, usize);

/// Front L/R and surround L/R core channel indices for sum/difference
/// decoding (Annex C.3.5), per `AMODE`.
fn sum_diff_pairs(amode: u32) -> (Option<Pair>, Option<Pair>) {
    match amode {
        1..=4 | 6 => (Some((0, 1)), None),
        5 | 7 => (Some((1, 2)), None),
        8 => (Some((0, 1)), Some((2, 3))),
        9 => (Some((1, 2)), Some((3, 4))),
        _ => (None, None),
    }
}

/// The core's speakers (coded order) for an `AMODE`, or the refusal.
fn core_speakers(amode: u32) -> Result<&'static [Speaker], Error> {
    layout::amode_speakers(amode).ok_or_else(|| {
        if amode <= 15 {
            Error::Unsupported(format!(
                "AMODE {amode} ({} channels) does not say which of its channels the core carries",
                AMODE_CHANNELS[amode as usize]
            ))
        } else {
            Error::Unsupported(format!("AMODE {amode} is a user-defined channel arrangement"))
        }
    })
}

/// The frame-level fields the decoder acts on (Tables 5-1 and 5-21).
struct FrameHeader {
    total_samples: usize,
    amode: u32,
    sample_rate: u32,
    lossless_steps: bool,
    dynf: bool,
    ext_audio_id: u32,
    ext_audio: bool,
    aspf: bool,
    /// `LFF`: 0 none, 1 → 128× interpolation, 2 → 64×.
    lff: u32,
    hflag: bool,
    filts_perfect: bool,
    sumf: bool,
    sums: bool,
    cpf: bool,
    subframes: usize,
    params: CodingParams,
}

/// Read the 14-bit `FSIZE` without a full parse, so the reader can be bounded
/// to the frame before anything else is decoded.
fn frame_len_of(buf: &[u8]) -> Result<usize, Error> {
    if buf.len() < 8 {
        return Err(Error::Truncated { at_bit: buf.len() * 8, wanted: 64 });
    }
    let mut r = BitReader::new(buf);
    let sync = r.bits(32)?;
    if sync != CORE_SYNC && sync != exss::SYNC_EXSS_CORE {
        return Err(Error::NoSync);
    }
    r.bits(1 + 5 + 1 + 7)?;
    Ok(r.bits(14)? as usize + 1)
}

fn parse_header(r: &mut BitReader) -> Result<FrameHeader, Error> {
    let sync = r.bits(32)?;
    if sync != CORE_SYNC && sync != exss::SYNC_EXSS_CORE {
        return Err(Error::NoSync);
    }
    let ftype = r.bits(1)?;
    let _short = r.bits(5)?;
    let cpf = r.flag()?;
    let nblks = r.bits(7)?;
    let _fsize = r.bits(14)?;
    let amode = r.bits(6)?;
    let sfreq = r.bits(4)?;
    let rate = r.bits(5)?;
    let _fixed = r.bits(1)?;
    let dynf = r.flag()?;
    let _timef = r.flag()?;
    let _auxf = r.flag()?;
    let _hdcd = r.flag()?;
    let ext_audio_id = r.bits(3)?;
    let ext_audio = r.flag()?;
    let aspf = r.flag()?;
    let lff = r.bits(2)?;
    let hflag = r.flag()?;
    if cpf {
        let _hcrc = r.bits(16)?;
    }
    let filts = r.flag()?;
    let vernum = r.bits(4)?;
    let _chist = r.bits(2)?;
    let _pcmr = r.bits(3)?;
    let sumf = r.flag()?;
    let sums = r.flag()?;
    let _dialnorm = r.bits(4)?;

    if ftype == 0 {
        return Err(Error::Unsupported(
            "termination frame (FTYPE = 0, partial subsubframes) — only normal frames are decoded".into(),
        ));
    }
    if nblks < 5 {
        return Err(Error::Invalid("NBLKS below 5"));
    }
    let sample_rate = SAMPLE_RATES[sfreq as usize];
    if sample_rate == 0 {
        return Err(Error::Invalid("reserved SFREQ code"));
    }
    if lff == 3 {
        return Err(Error::Invalid("LFF = 3"));
    }
    if vernum > 7 {
        return Err(Error::Unsupported(format!(
            "VERNUM {vernum}: encoder revision incompatible with ETSI TS 102 114 V1.6.1 (spec says mute)"
        )));
    }
    core_speakers(amode)?;

    // Primary audio coding header (Table 5-21).
    let subframes = r.bits(4)? as usize + 1;
    let channels = r.bits(3)? as usize + 1;
    if channels > MAX_CHANNELS {
        return Err(Error::Invalid("more than five primary channels in the core"));
    }
    if channels != AMODE_CHANNELS[amode as usize] {
        return Err(Error::Invalid("PCHS disagrees with AMODE"));
    }
    let params = core::parse_coding_params(r, channels, &[])?;
    if cpf {
        let _ahcrc = r.bits(16)?;
    }
    Ok(FrameHeader {
        total_samples: 32 * (nblks as usize + 1),
        amode,
        sample_rate,
        lossless_steps: rate == 0x1F,
        dynf,
        ext_audio_id,
        ext_audio,
        aspf,
        lff,
        hflag,
        filts_perfect: filts,
        sumf,
        sums,
        cpf,
        subframes,
        params,
    })
}

/// Which extensions a frame carried: decoded ([`CoreInfo::extensions`]) or
/// present but not decoded ([`CoreInfo::skipped`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Extensions {
    /// XCh: the 6.1 back centre (6.4).
    pub xch: bool,
    /// XXCH: channels beyond 5.1 (6.5).
    pub xxch: bool,
    /// X96: the band above the core rate (6.2).
    pub x96: bool,
    /// XBR: extended bit-rate residuals (6.3).
    pub xbr: bool,
    /// A DTS-HD extension substream (clause 7).
    pub exss: bool,
    /// XLL: the lossless extension (clause 8).
    pub xll: bool,
    /// LBR: the low bit-rate extension (clause 9).
    pub lbr: bool,
}

impl Extensions {
    fn any(&self) -> bool {
        self.xch || self.xxch || self.x96 || self.xbr || self.exss || self.xll || self.lbr
    }
}

/// One decoded frame's PCM.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Interleaved f32 at ±1.0 full scale, `channels` per sample frame, in
    /// [`layout`](Self::layout)'s speaker order: 32 × (`NBLKS` + 1) samples
    /// per channel (512 for the common 48 kHz frame), twice that with X96.
    pub samples: Vec<f32>,
    /// The output sample rate: the core's (8–48 kHz), or twice it when the
    /// X96 extension was decoded.
    pub sample_rate: u32,
    /// Channels, LFE included: `layout.channels()`.
    pub channels: usize,
    /// What each interleaved slot carries.
    pub layout: Layout,
}

impl Frame {
    /// Samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.channels
    }
}

/// What the header of the last frame said, for a caller's diagnostics: the
/// fields that decide the output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CoreInfo {
    /// `AMODE`, the core's channel arrangement (Table 5-4).
    pub amode: u8,
    /// Whether an LFE channel is coded (`LFF` ≠ 0).
    pub lfe: bool,
    /// The layout the frame decodes to (extensions included).
    pub layout: Layout,
    /// The output sample rate (the core's `SFREQ`, Table 5-5, doubled by X96).
    pub sample_rate: u32,
    /// The core's own sample rate.
    pub core_sample_rate: u32,
    /// `FILTS`: the perfect-reconstruction QMF prototype (`true`) or the
    /// non-perfect one.
    pub perfect_reconstruction: bool,
    /// The extensions this frame's output includes.
    pub extensions: Extensions,
    /// Extensions the frame carried that are not in its output (XLL, LBR;
    /// everything when [`Decoder::set_core_only`] is on).
    pub skipped: Extensions,
}

/// `FSIZE + 1` of the core frame at the start of `buf`: its length in
/// bytes, read without decoding it. `buf` must start with [`CORE_SYNC`].
/// A DTS-HD stream follows each core frame with extension substream
/// frames; [`packet_len`] covers those too.
pub fn frame_len(buf: &[u8]) -> Result<usize> {
    frame_len_of(buf)
}

/// Up to three zero bytes of DWORD padding (7.5.1) at `off`.
fn skip_padding(buf: &[u8], mut off: usize) -> usize {
    let mut n = 0;
    while n < 3 && off < buf.len() && buf[off] == 0 && !buf[off..].starts_with(&exss::SYNC_EXSS.to_be_bytes()) {
        off += 1;
        n += 1;
    }
    off
}

fn be32(buf: &[u8], off: usize) -> Option<u32> {
    buf.get(off..off + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// The length of the DTS packet (one access unit) at the start of `buf`: a
/// core frame plus any extension substream frames (and padding) that
/// follow it, or a run of extension substream frames on their own. For
/// splitting a raw `.dts` / `.dtshd` elementary stream into the packets
/// [`Decoder::decode`] takes.
pub fn packet_len(buf: &[u8]) -> Result<usize> {
    let mut off = match be32(buf, 0) {
        Some(CORE_SYNC) => frame_len_of(buf)?,
        Some(exss::SYNC_EXSS) => 0,
        _ => return Err(Error::NoSync),
    };
    let mut seen_index = None;
    loop {
        let at = skip_padding(buf, off);
        if be32(buf, at) != Some(exss::SYNC_EXSS) {
            break;
        }
        // A second substream frame with an index not above the last one
        // belongs to the next access unit.
        let index = buf.get(at + 5).map(|b| b >> 6).unwrap_or(0);
        if seen_index.is_some_and(|i| index <= i) {
            break;
        }
        seen_index = Some(index);
        off = at + exss::frame_size(&buf[at..])?;
    }
    Ok(off.min(buf.len()).max(4))
}

/// Re-frame a raw DTS stream to the 16-bit big-endian framing the decoder
/// reads. Raw `.dts`/`.wav` (DTS-CD) streams also come as 16-bit
/// little-endian words and as 14-bit words (in either byte order), each
/// carrying 14 bits of the stream (sync words 0xFE7F0180, 0x1FFFE800,
/// 0xFF1F00E8). Returns the input unchanged when it is already 16-bit
/// big-endian.
pub fn normalize_framing(raw: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>> {
    use std::borrow::Cow;
    let find = |pat: &[u8]| raw.windows(pat.len()).position(|w| w == pat);
    let be16 = find(&[0x7F, 0xFE, 0x80, 0x01]);
    let le16 = find(&[0xFE, 0x7F, 0x01, 0x80]);
    let be14 = find(&[0x1F, 0xFF, 0xE8, 0x00, 0x07]);
    let le14 = find(&[0xFF, 0x1F, 0x00, 0xE8]);
    let first = [be16, le16, be14, le14].iter().enumerate().filter_map(|(i, p)| p.map(|p| (p, i))).min();
    let Some((start, kind)) = first else {
        return Err(Error::NoSync);
    };
    match kind {
        0 => Ok(Cow::Borrowed(raw)),
        1 => Ok(Cow::Owned(raw[start..].as_chunks::<2>().0.iter().flat_map(|w| [w[1], w[0]]).collect())),
        _ => {
            // 14-bit: words of 14 payload bits. Split at each sync (three
            // words: 0x1FFF 0xE800 0x07Fx) and repack each frame's words
            // into bytes, so every frame starts byte-aligned.
            let le = kind == 3;
            let words: Vec<u16> = raw[start..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| if le { u16::from_le_bytes([w[0], w[1]]) } else { u16::from_be_bytes([w[0], w[1]]) })
                .collect();
            let mut out = Vec::with_capacity(words.len() * 14 / 8 + 8);
            let mut i = 0;
            while i < words.len() {
                let mut j = i + 3;
                while j + 2 < words.len()
                    && !(words[j] == 0x1FFF && words[j + 1] == 0xE800 && words[j + 2] & 0x3FF0 == 0x07F0)
                {
                    j += 1;
                }
                let end = if j + 2 < words.len() { j } else { words.len() };
                let (mut acc, mut nbits) = (0u32, 0u32);
                for &w in &words[i..end] {
                    acc = (acc << 14) | (w as u32 & 0x3FFF);
                    nbits += 14;
                    while nbits >= 8 {
                        out.push((acc >> (nbits - 8)) as u8);
                        nbits -= 8;
                    }
                    acc &= (1 << nbits) - 1;
                }
                if nbits > 0 {
                    out.push((acc << (8 - nbits)) as u8);
                }
                i = end;
            }
            Ok(Cow::Owned(out))
        }
    }
}

/// One channel's synthesis and prediction state, kept across frames.
struct ChState {
    qmf: Qmf,
    qmf64: Qmf64,
    pred: adpcm::PredictorState,
    pred96: adpcm::PredictorState,
}

impl ChState {
    fn new() -> Self {
        Self {
            qmf: Qmf::new(),
            qmf64: Qmf64::new(),
            pred: adpcm::PredictorState::new(NSB),
            pred96: adpcm::PredictorState::new(64),
        }
    }
}

/// A decoded full-band channel before synthesis.
struct Channel {
    speaker: Speaker,
    /// 32 or 64 subbands × blocks.
    buf: ChannelBuf,
    /// The set's `SHUFF` for this channel and its `TMODE` per subframe
    /// (what XBR reads its scale factors with).
    shuff: u32,
    tmode: Vec<[u32; NSB]>,
    /// Index into the decoder's state: `Core(i)` or `Ext(i)`.
    state: StateId,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StateId {
    Core(usize),
    Ext(usize),
}

/// A DTS decoder. One per stream: the filter banks and predictors keep
/// history from frame to frame.
pub struct Decoder {
    core: Vec<ChState>,
    ext: Vec<ChState>,
    lfe: LfeInterp,
    lfe2x: Lfe2x,
    noise: ext::Noise,
    adpcm_book: Option<Arc<AdpcmCodebook>>,
    hf_book: Option<Arc<HfVqCodebook>>,
    fallback: AdpcmFallback,
    core_only: bool,
    hf_vq_skipped: bool,
    /// The header of the last frame whose layout was mapped.
    info: Option<CoreInfo>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder that decodes every extension it can, refuses predicted
    /// frames it has no code book for, and decodes VQ subbands as silence
    /// until given the D.10.2 code book.
    pub fn new() -> Self {
        Self {
            core: (0..MAX_CHANNELS).map(|_| ChState::new()).collect(),
            ext: Vec::new(),
            lfe: LfeInterp::new(),
            lfe2x: Lfe2x::default(),
            noise: ext::Noise::default(),
            adpcm_book: None,
            hf_book: None,
            fallback: AdpcmFallback::Refuse,
            core_only: false,
            hf_vq_skipped: false,
            info: None,
        }
    }

    /// Supply the D.10.1 ADPCM code book (see [`vq`]): predicted subbands
    /// then decode exactly.
    pub fn set_adpcm_codebook(&mut self, book: Option<Arc<AdpcmCodebook>>) {
        self.adpcm_book = book;
    }

    /// Supply the D.10.2 high-frequency VQ code book (see [`vq`]).
    pub fn set_hf_vq_codebook(&mut self, book: Option<Arc<HfVqCodebook>>) {
        self.hf_book = book;
    }

    /// What to do with predicted subbands when no ADPCM code book was
    /// supplied (default [`AdpcmFallback::Refuse`]).
    pub fn set_adpcm_fallback(&mut self, fallback: AdpcmFallback) {
        self.fallback = fallback;
    }

    /// Decode the core only, as a legacy DTS decoder does: XCh, XXCH, X96,
    /// XBR and the extension substream are skipped (and reported in
    /// [`CoreInfo::skipped`]). Off by default.
    pub fn set_core_only(&mut self, core_only: bool) {
        self.core_only = core_only;
    }

    /// Decode a packet: one or more core frames back to back, as Matroska
    /// `A_DTS`, MP4 `dtsc`/`dtsh`/`dtsl` and a raw stream split by
    /// [`packet_len`] carry them, each optionally followed by DTS-HD
    /// extension substream frames; or extension substream frames alone when
    /// they carry a core. Every frame decodes to its full block of PCM as
    /// it arrives, so nothing is held back and there is nothing to flush.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<Frame>> {
        let mut frames = Vec::new();
        let mut off = 0usize;
        while off + 4 <= packet.len() {
            let sync = be32(packet, off).expect("4 bytes");
            if sync == CORE_SYNC {
                let flen = frame_len_of(&packet[off..])?;
                if packet.len() - off < flen {
                    return Err(Error::Truncated {
                        at_bit: (packet.len() - off) * 8,
                        wanted: ((flen - (packet.len() - off)) * 8) as u32,
                    });
                }
                let mut next = off + flen;
                let mut exss_frames = Vec::new();
                loop {
                    let at = skip_padding(packet, next);
                    if be32(packet, at) != Some(exss::SYNC_EXSS) {
                        break;
                    }
                    let size = exss::frame_size(&packet[at..])?;
                    if at + size > packet.len() {
                        return Err(Error::Truncated { at_bit: packet.len() * 8, wanted: ((at + size - packet.len()) * 8) as u32 });
                    }
                    exss_frames.push(&packet[at..at + size]);
                    next = at + size;
                }
                frames.push(self.decode_frame(&packet[off..off + flen], &exss_frames)?);
                off = next;
            } else if sync == exss::SYNC_EXSS && off == 0 {
                let size = exss::frame_size(packet)?;
                if size > packet.len() {
                    return Err(Error::Truncated { at_bit: packet.len() * 8, wanted: ((size - packet.len()) * 8) as u32 });
                }
                let f = exss::parse(&packet[..size])?;
                let core_range = f
                    .assets
                    .first()
                    .and_then(|a| a.component(exss::mask::EXSS_CORE))
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "DTS-HD extension substream without a core component (lossless-only or LBR stream)"
                                .into(),
                        )
                    })?;
                let core_bytes = &packet[core_range];
                let flen = frame_len_of(core_bytes)?.min(core_bytes.len());
                frames.push(self.decode_frame(&core_bytes[..flen], &[&packet[..size]])?);
                off = size;
            } else if off == 0 {
                return Err(Error::NoSync);
            } else {
                // Padding or something else follows: there is no second
                // frame behind it.
                break;
            }
        }
        Ok(frames)
    }

    /// The layout of the last frame whose header was read (set before its
    /// audio is decoded, so a frame refused later still sets it).
    pub fn layout(&self) -> Option<Layout> {
        self.info.map(|i| i.layout)
    }

    /// The header fields of the last frame whose layout was mapped; `None`
    /// before the first.
    pub fn info(&self) -> Option<CoreInfo> {
        self.info
    }

    /// Whether any frame so far carried high-frequency VQ subbands that
    /// decoded as silence for want of the D.10.2 code book (§5.4.3 allows
    /// ignoring them). Sticky: once true, it stays true.
    pub fn hf_vq_skipped(&self) -> bool {
        self.hf_vq_skipped
    }

    /// How many subband predictions so far were estimated
    /// ([`AdpcmFallback::Estimate`]) rather than decoded.
    pub fn adpcm_estimated(&self) -> u64 {
        self.core.iter().chain(&self.ext).map(|s| s.pred.estimated + s.pred96.estimated).sum()
    }

    fn ext_state(&mut self, i: usize) -> &mut ChState {
        while self.ext.len() <= i {
            self.ext.push(ChState::new());
        }
        &mut self.ext[i]
    }

    /// Decode one core frame (`core`, exactly `FSIZE + 1` bytes) with the
    /// extension substream frames that followed it.
    fn decode_frame(&mut self, core: &[u8], exss_frames: &[&[u8]]) -> Result<Frame, Error> {
        let mut r = BitReader::new(core);
        let h = parse_header(&mut r)?;
        let core_spk = core_speakers(h.amode)?;
        let mut speakers: Vec<Speaker> = core_spk.to_vec();
        if h.lff > 0 {
            speakers.push(Speaker::LFE);
        }
        let mut info = CoreInfo {
            amode: h.amode as u8,
            lfe: h.lff > 0,
            layout: Layout::from_speakers(&speakers),
            sample_rate: h.sample_rate,
            core_sample_rate: h.sample_rate,
            perfect_reconstruction: h.filts_perfect,
            extensions: Extensions::default(),
            skipped: Extensions::default(),
        };
        self.info = Some(info);

        let p = &h.params;
        let n = p.n;
        let blocks = h.total_samples / 32;
        let step_table: &[u32; 27] =
            if h.lossless_steps { &tables::STEP_SIZE_LOSSLESS_Q22 } else { &tables::STEP_SIZE_LOSSY_Q22 };
        let (front_pair, surround_pair) = sum_diff_pairs(h.amode);
        let adpcm_book = self.adpcm_book.clone();
        let hf_book = self.hf_book.clone();
        let ctx = AudioCtx {
            step_table,
            aspf: h.aspf,
            adpcm_book: adpcm_book.as_deref(),
            hf_book: hf_book.as_deref(),
            fallback: self.fallback,
        };

        let mut bufs: Vec<ChannelBuf> = (0..n).map(|_| ChannelBuf::new(NSB, blocks)).collect();
        let mut pred: Vec<adpcm::PredictorState> = self.core[..n].iter().map(|s| s.pred.clone()).collect();
        for pr in &mut pred {
            pr.begin_frame(h.hflag);
        }
        let mut cc = CoreFrameCtx {
            ssc: Vec::with_capacity(h.subframes),
            aspf: h.aspf,
            cpf: h.cpf,
            lossless_steps: h.lossless_steps,
        };
        let mut core_tmode: Vec<Vec<[u32; NSB]>> = vec![Vec::with_capacity(h.subframes); n];
        let mut range = vec![1.0f64; blocks];
        let mut lfe_dec: Vec<f64> = Vec::new();
        let mut hf_skipped = false;
        let mut t0 = 0usize;
        for _ in 0..h.subframes {
            let ssc = r.bits(2)? as usize + 1;
            let psc = r.bits(3)?;
            if psc != 0 {
                return Err(Error::Invalid("partial subsubframe (PSC) in a normal frame"));
            }
            let si = core::parse_side_info(&mut r, p, ssc)?;
            let rng = if h.dynf { tables::DRC_MULTIPLIER[r.bits(8)? as usize] as f64 } else { 1.0 };
            if h.cpf {
                let _sicrc = r.bits(16)?;
            }
            ext::check_prediction(si.predicted(p), &ctx, "")?;
            if t0 + 8 * ssc > blocks {
                return Err(Error::Invalid("subframes do not add up to NBLKS blocks"));
            }
            core::read_hf_vq(&mut r, p, &si, t0, &mut bufs, ctx.hf_book, &mut hf_skipped)?;

            // LFE: 2·LFF·nSSC 8-bit samples, then a 7-bit-table scale index.
            if h.lff > 0 {
                let count = 2 * h.lff as usize * ssc;
                let mut raw = Vec::with_capacity(count);
                for _ in 0..count {
                    raw.push(r.sbits(8)? as f64);
                }
                let idx = r.bits(8)? as usize;
                let scale = tables::SCALE_RMS_7BIT
                    .get(idx)
                    .copied()
                    .filter(|v| *v > 0)
                    .ok_or(Error::Invalid("LFE scale index outside the 7-bit table"))?;
                let rscale = scale as f64 * 0.035;
                lfe_dec.extend(raw.iter().map(|v| v * rscale));
            }

            for ssf in 0..ssc {
                core::read_subsubframe(&mut r, p, &si, ssf, t0 + 8 * ssf, &ctx, &mut bufs, &mut pred)?;
            }
            core::finish_estimates(p, &si, t0, 8 * ssc, &ctx, &mut bufs, &mut pred);
            core::apply_joint(p, &si, &mut bufs, &[], t0, 8 * ssc);
            // Sum/difference decoding (Annex C.3.5); AMODE 3 is coded that
            // way by definition.
            let mut pairs = Vec::new();
            if h.sumf || h.amode == 3 {
                pairs.extend(front_pair);
            }
            if h.sums {
                pairs.extend(surround_pair);
            }
            for (l, rr) in pairs {
                for sb in 0..NSB {
                    for s in t0..t0 + 8 * ssc {
                        let a = bufs[l].s[sb][s];
                        let b = bufs[rr].s[sb][s];
                        bufs[l].s[sb][s] = a + b;
                        bufs[rr].s[sb][s] = a - b;
                    }
                }
            }
            range[t0..t0 + 8 * ssc].fill(rng);
            cc.ssc.push(ssc);
            for (ch, t) in core_tmode.iter_mut().enumerate() {
                t.push(si.tmode[ch]);
            }
            t0 += 8 * ssc;
        }
        if t0 != blocks {
            return Err(Error::Invalid("subframes do not add up to NBLKS blocks"));
        }
        if h.lff > 0 && lfe_dec.len() * if h.lff == 1 { 128 } else { 64 } != h.total_samples {
            return Err(Error::Invalid("LFE samples do not add up to NBLKS blocks"));
        }
        let audio_end = r.position_bits().div_ceil(8);

        // Full-band channels, core first.
        let mut channels: Vec<Channel> = bufs
            .into_iter()
            .zip(core_tmode)
            .enumerate()
            .map(|(i, (buf, tmode))| Channel { speaker: core_spk[i], buf, shuff: p.shuff[i], tmode, state: StateId::Core(i) })
            .collect();
        let mut ext_pred: Vec<(usize, adpcm::PredictorState)> = Vec::new();
        let mut x96: Option<(Vec<ChannelBuf>, Vec<adpcm::PredictorState>)> = None;
        // Undo XCh/XXCH embedded downmixes after synthesis: (target channel
        // indices with their gains, source channel index, scale).
        let mut downmixes: Vec<Downmix> = Vec::new();
        let mut core_lfe_mask_ok = true;

        let mut found = Extensions::default();
        // Extensions inside the core frame (EXT_AUDIO), DWORD-aligned
        // from the frame start.
        let mut in_core: Vec<(u32, usize)> = Vec::new();
        if h.ext_audio {
            let mut at = audio_end.div_ceil(4) * 4;
            while at + 4 <= core.len() {
                match be32(core, at) {
                    Some(ext::SYNC_XCH) if matches!(h.ext_audio_id, 0 | 3) => {
                        let fsize = BitReader::new(&core[at + 4..]).bits(10)? as usize;
                        let dist = core.len() - at;
                        // XChFSIZE + 1 is the distance to the frame end; legacy
                        // streams use XChFSIZE (6.4.2). An exact match wins over
                        // a legacy one (data can alias the sync just before it).
                        let exact = dist == fsize + 1;
                        if exact || dist == fsize {
                            let prev = in_core.iter().position(|(s, _)| *s == ext::SYNC_XCH);
                            match prev {
                                Some(i) if exact => in_core[i] = (ext::SYNC_XCH, at),
                                Some(_) => {}
                                None => in_core.push((ext::SYNC_XCH, at)),
                            }
                            found.xch = true;
                        }
                    }
                    Some(ext::SYNC_X96) if matches!(h.ext_audio_id, 2 | 3) => {
                        let fsize = BitReader::new(&core[at + 4..]).bits(12)? as usize + 1;
                        if fsize >= 4 && at + fsize <= core.len() {
                            in_core.push((ext::SYNC_X96, at));
                            found.x96 = true;
                        }
                    }
                    Some(ext::SYNC_XXCH) if h.ext_audio_id == 6 && ext::parse_xxch_header(&core[at..]).is_ok() => {
                        in_core.push((ext::SYNC_XXCH, at));
                        found.xxch = true;
                    }
                    _ => {}
                }
                at += 4;
            }
        }
        let mut parsed_exss = Vec::new();
        for f in exss_frames {
            found.exss = true;
            let fr = exss::parse(f)?;
            for a in &fr.assets {
                found.xll |= a.ext_mask & exss::mask::EXSS_XLL != 0;
                found.lbr |= a.ext_mask & exss::mask::EXSS_LBR != 0;
                found.xbr |= a.ext_mask & exss::mask::EXSS_XBR != 0;
                found.xxch |= a.ext_mask & exss::mask::EXSS_XXCH != 0;
                found.x96 |= a.ext_mask & exss::mask::EXSS_X96 != 0;
            }
            parsed_exss.push((f, fr));
        }
        // Asset 0 of extension substream 0 extends the core.
        let asset0 = parsed_exss
            .iter()
            .find(|(_, fr)| fr.index == 0)
            .and_then(|(f, fr)| fr.assets.first().filter(|a| a.coding_mode == 0).map(|a| (*f, a.clone())));

        let mut used = Extensions::default();
        if !self.core_only {
            used.exss = found.exss;
            let ext_ctx = &ctx;
            // XCh.
            if let Some(&(_, at)) = in_core.iter().find(|(s, _)| *s == ext::SYNC_XCH) {
                let mut xr = BitReader::new(&core[at..]);
                let lower_subs: Vec<usize> = (0..n).map(|i| p.subs[i]).collect();
                let (_fsize, xp) = ext::parse_xch_header(&mut xr, h.cpf, &lower_subs)?;
                let mut xb: Vec<ChannelBuf> = (0..xp.n).map(|_| ChannelBuf::new(NSB, blocks)).collect();
                let base = 0;
                let mut xpred: Vec<adpcm::PredictorState> =
                    (0..xp.n).map(|i| self.ext_state(base + i).pred.clone()).collect();
                for pr in &mut xpred {
                    pr.begin_frame(h.hflag);
                }
                let lower: Vec<&ChannelBuf> = channels.iter().map(|c| &c.buf).collect();
                let tmodes =
                    ext::decode_set_subframes(&mut xr, &xp, &lower, &cc, ext_ctx, &mut xb, &mut xpred, &mut hf_skipped, " of XCh")?;
                if xp.n != 1 {
                    return Err(Error::Unsupported(format!("XCh with {} channels (only the back centre is defined)", xp.n)));
                }
                let src = channels.len();
                for (i, (buf, pr)) in xb.into_iter().zip(xpred).enumerate() {
                    let tmode = tmodes.iter().map(|t| t[i]).collect();
                    channels.push(Channel { speaker: Speaker::BC, buf, shuff: xp.shuff[i], tmode, state: StateId::Ext(base + i) });
                    ext_pred.push((base + i, pr));
                }
                let targets: Vec<(usize, f64)> = channels
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| matches!(c.speaker, Speaker::SL | Speaker::SR) && matches!(c.state, StateId::Core(_)))
                    .map(|(i, _)| (i, XCH_DOWNMIX))
                    .collect();
                downmixes.push(Downmix { source: src, targets, scale: None });
                used.xch = true;
            }
            // XXCH, in the core or in asset 0.
            let xxch_bytes: Option<&[u8]> = in_core
                .iter()
                .find(|(s, _)| *s == ext::SYNC_XXCH)
                .map(|&(_, at)| &core[at..])
                .or_else(|| {
                    asset0.as_ref().and_then(|(f, a)| a.component(exss::mask::EXSS_XXCH).map(|rg| &f[rg]))
                });
            if let Some(data) = xxch_bytes {
                let xh = ext::parse_xxch_header(data)?;
                // The core's speakers by the XXCH core activity mask.
                let core_bits: Vec<u32> = (0..32).filter(|b| (xh.core_mask >> b) & 1 == 1 && *b != 5).collect();
                if core_bits.len() == n {
                    for (c, b) in channels.iter_mut().zip(&core_bits) {
                        c.speaker = layout::xxch_mask_speaker(*b).unwrap_or(c.speaker);
                    }
                } else {
                    core_lfe_mask_ok = false;
                }
                let mut next_state = channels.iter().filter(|c| matches!(c.state, StateId::Ext(_))).count();
                let mut lower_subs: Vec<usize> = (0..n).map(|i| p.subs[i]).collect();
                lower_subs.extend(std::iter::repeat_n(0, channels.len() - n));
                for &(s0, s1) in &xh.sets {
                    let set_bytes = &data[s0..s1];
                    let mut xr = BitReader::new(set_bytes);
                    let (set, hsize) = ext::parse_xxch_set_header(&mut xr, set_bytes, &xh, &lower_subs)?;
                    xr.seek_bits(hsize * 8)?;
                    let k = set.params.n;
                    let mut xb: Vec<ChannelBuf> = (0..k).map(|_| ChannelBuf::new(NSB, blocks)).collect();
                    let mut xpred: Vec<adpcm::PredictorState> =
                        (0..k).map(|i| self.ext_state(next_state + i).pred.clone()).collect();
                    for pr in &mut xpred {
                        pr.begin_frame(h.hflag);
                    }
                    let lower: Vec<&ChannelBuf> = channels.iter().map(|c| &c.buf).collect();
                    let tmodes = ext::decode_set_subframes(
                        &mut xr,
                        &set.params,
                        &lower,
                        &cc,
                        ext_ctx,
                        &mut xb,
                        &mut xpred,
                        &mut hf_skipped,
                        " of XXCH",
                    )?;
                    lower_subs.extend_from_slice(&set.params.subs[..k]);
                    let first = channels.len();
                    for (i, (buf, pr)) in xb.into_iter().zip(xpred).enumerate() {
                        let speaker = layout::xxch_mask_speaker(set.channel_bits[i])
                            .ok_or(Error::Invalid("XXCH speaker mask bit with no position"))?;
                        let tmode = tmodes.iter().map(|t| t[i]).collect();
                        let shuff = set.params.shuff[i];
                        channels.push(Channel { speaker, buf, shuff, tmode, state: StateId::Ext(next_state + i) });
                        ext_pred.push((next_state + i, pr));
                    }
                    if let Some((coeffs, scale)) = &set.downmix {
                        for (i, c) in coeffs.iter().enumerate() {
                            let mut targets = Vec::new();
                            for &(bit, gain) in c {
                                let spk = layout::xxch_mask_speaker(bit);
                                if let Some(t) = channels[..first].iter().position(|ch| Some(ch.speaker) == spk) {
                                    targets.push((t, gain));
                                }
                            }
                            downmixes.push(Downmix { source: first + i, targets, scale: (i == 0).then_some((*scale, first)) });
                        }
                    }
                    next_state += k;
                }
                used.xxch = true;
            }
            // X96, in the core or in asset 0: 64-band residuals, then the
            // core's subband samples added into the lower 32 bands.
            let x96_core = in_core.iter().find(|(s, _)| *s == ext::SYNC_X96).map(|&(_, at)| &core[at..]);
            let x96_exss = asset0.as_ref().and_then(|(f, a)| a.component(exss::mask::EXSS_X96).map(|rg| &f[rg]));
            if let Some(data) = x96_exss.or(x96_core) {
                let in_exss = x96_exss.is_some();
                let mut xr = BitReader::new(data);
                if xr.bits(32)? != ext::SYNC_X96 {
                    return Err(Error::Invalid("X96 sync word"));
                }
                let mut b64: Vec<ChannelBuf> = channels.iter().map(|_| ChannelBuf::new(64, blocks)).collect();
                let mut p96: Vec<adpcm::PredictorState> = channels
                    .iter()
                    .map(|c| match c.state {
                        StateId::Core(i) => self.core[i].pred96.clone(),
                        StateId::Ext(i) => self.ext_state(i).pred96.clone(),
                    })
                    .collect();
                for pr in &mut p96 {
                    pr.begin_frame(h.hflag);
                }
                if in_exss {
                    let header_size = xr.bits(6)? as usize + 1;
                    let revno = xr.bits(4)?;
                    let crc_chset = xr.flag()?;
                    let nsets = xr.bits(2)? as usize + 1;
                    let mut sizes = Vec::with_capacity(nsets);
                    for _ in 0..nsets {
                        sizes.push(xr.bits(12)? as usize + 1);
                    }
                    let mut nch = Vec::with_capacity(nsets);
                    for _ in 0..nsets {
                        nch.push(xr.bits(3)? as usize + 1);
                    }
                    if header_size > data.len() || crc::crc16(&data[4..header_size]) != 0 {
                        return Err(Error::Invalid("X96 header CRC"));
                    }
                    if revno > 8 {
                        return Err(Error::Unsupported(format!("X96 REVNO {revno} (above 8: ignore, 6.2.4.2)")));
                    }
                    let (mut off, mut ch0) = (header_size, 0usize);
                    for (set, &size) in sizes.iter().enumerate() {
                        let k = nch[set];
                        if off + size > data.len() || ch0 + k > channels.len() {
                            break;
                        }
                        let bytes = &data[off..off + size];
                        let mut sr = BitReader::new(bytes);
                        let hsize = sr.bits(7)? as usize + 1;
                        let s = ext::parse_x96_set_header(&mut sr, k, revno, true)?;
                        if crc_chset && crc::crc16(&bytes[..hsize]) != 0 {
                            return Err(Error::Invalid("X96 channel set header CRC"));
                        }
                        sr.seek_bits(hsize * 8)?;
                        ext::decode_x96_subframes(
                            &mut sr,
                            &s,
                            &cc,
                            ext_ctx,
                            &mut b64[ch0..ch0 + k],
                            &mut p96[ch0..ch0 + k],
                            &mut self.noise,
                            &mut hf_skipped,
                        )?;
                        off += size;
                        ch0 += k;
                    }
                } else {
                    let _fsize96 = xr.bits(12)?;
                    let revno = xr.bits(4)?;
                    if revno > 8 {
                        return Err(Error::Unsupported(format!("X96 REVNO {revno} (above 8: ignore, 6.2.4.2)")));
                    }
                    let s = ext::parse_x96_set_header(&mut xr, n, revno, false)?;
                    if h.cpf {
                        let _ahcrc96 = xr.bits(16)?;
                    }
                    ext::decode_x96_subframes(&mut xr, &s, &cc, ext_ctx, &mut b64[..n], &mut p96[..n], &mut self.noise, &mut hf_skipped)?;
                }
                for (c, b) in channels.iter().zip(b64.iter_mut()) {
                    for sb in 0..NSB {
                        for (o, v) in b.s[sb].iter_mut().zip(&c.buf.s[sb]) {
                            *o += v;
                        }
                    }
                }
                x96 = Some((b64, p96));
                used.x96 = true;
            }
            // XBR from asset 0: residuals into the channel sets, in order
            // (core, then XCh/XXCH channels).
            if let Some((f, a)) = &asset0
                && let Some(rg) = a.component(exss::mask::EXSS_XBR)
            {
                let mut list: Vec<ext::XbrTarget> = match &mut x96 {
                    Some((b64, _)) => b64
                        .iter_mut()
                        .zip(&channels)
                        .map(|(buf, c)| ext::XbrTarget { buf, shuff: c.shuff, tmode: &c.tmode })
                        .collect(),
                    None => channels
                        .iter_mut()
                        .map(|c| ext::XbrTarget { buf: &mut c.buf, shuff: c.shuff, tmode: &c.tmode })
                        .collect(),
                };
                // Channel set 0 is the core's; the next ones the extension
                // channels in order.
                let rest = list.split_off(n.min(list.len()));
                let mut targets = vec![list, rest];
                ext::decode_xbr(&f[rg], &cc, &mut targets)?;
                used.xbr = true;
            }
        }
        let mut skipped = Extensions {
            xch: found.xch && !used.xch,
            xxch: found.xxch && !used.xxch,
            x96: found.x96 && !used.x96,
            xbr: found.xbr && !used.xbr,
            exss: found.exss && !used.exss,
            xll: found.xll,
            lbr: found.lbr,
        };
        if !core_lfe_mask_ok {
            skipped.xxch = true;
        }

        // Output speakers: every channel once.
        let mut spk: Vec<Speaker> = channels.iter().map(|c| c.speaker).collect();
        if h.lff > 0 {
            spk.push(Speaker::LFE);
        }
        let mut sorted = spk.clone();
        sorted.sort();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::Unsupported(format!("two channels for the same speaker ({spk:?})")));
        }
        let layout = Layout::from_speakers(&spk);

        // Everything parsed: commit predictor state and synthesise.
        self.hf_vq_skipped |= hf_skipped;
        for (i, pr) in pred.into_iter().enumerate() {
            self.core[i].pred = pr;
        }
        for (i, pr) in ext_pred {
            self.ext_state(i).pred = pr;
        }
        for c in &channels {
            let st = match c.state {
                StateId::Core(i) => &mut self.core[i],
                StateId::Ext(i) => &mut self.ext[i],
            };
            st.pred.end_frame(&c.buf.s);
        }
        let out_rate;
        let samples_out;
        let mut pcm: Vec<Vec<f64>> = Vec::with_capacity(channels.len());
        if let Some((b64, p96)) = x96 {
            out_rate = 2 * h.sample_rate;
            samples_out = 2 * h.total_samples;
            for ((c, b), pr) in channels.iter().zip(&b64).zip(p96) {
                let st = match c.state {
                    StateId::Core(i) => &mut self.core[i],
                    StateId::Ext(i) => &mut self.ext[i],
                };
                let mut pr = pr;
                pr.end_frame(&b.s);
                st.pred96 = pr;
                let mut out = Vec::with_capacity(samples_out);
                let mut o64 = [0.0f64; 64];
                for t in 0..blocks {
                    let xin: [f64; 64] = std::array::from_fn(|sb| b.s[sb][t]);
                    st.qmf64.synthesize(&xin, &mut o64);
                    out.extend(o64.iter().map(|v| v * range[t]));
                }
                pcm.push(out);
            }
        } else {
            out_rate = h.sample_rate;
            samples_out = h.total_samples;
            for c in &channels {
                let st = match c.state {
                    StateId::Core(i) => &mut self.core[i],
                    StateId::Ext(i) => &mut self.ext[i],
                };
                let mut out = Vec::with_capacity(samples_out);
                let mut o32 = [0.0f64; 32];
                for t in 0..blocks {
                    let xin: [f64; 32] = std::array::from_fn(|sb| c.buf.s[sb][t]);
                    st.qmf.synthesize(&xin, h.filts_perfect, &mut o32);
                    out.extend(o32.iter().map(|v| v * range[t]));
                }
                pcm.push(out);
            }
        }
        let mut lfe_pcm = Vec::new();
        if h.lff > 0 {
            let factor = if h.lff == 1 { 128 } else { 64 };
            let mut core_lfe = Vec::with_capacity(h.total_samples);
            self.lfe.interpolate(&lfe_dec, factor, &mut core_lfe);
            if out_rate != h.sample_rate {
                self.lfe2x.interpolate(&core_lfe, &mut lfe_pcm);
            } else {
                lfe_pcm = core_lfe;
            }
        }
        // Undo embedded downmixes, in the order the sets were decoded.
        for d in &downmixes {
            if let Some((scale, upto)) = d.scale {
                for ch in pcm.iter_mut().take(upto) {
                    for v in ch.iter_mut() {
                        *v /= scale;
                    }
                }
            }
            let src = pcm[d.source].clone();
            for &(t, gain) in &d.targets {
                for (v, s) in pcm[t].iter_mut().zip(&src) {
                    *v -= gain * s;
                }
            }
        }

        let nout = spk.len();
        let mut out = vec![0.0f32; samples_out * nout];
        for (slot, s) in layout.speakers().iter().enumerate() {
            let src: &[f64] = if *s == Speaker::LFE && h.lff > 0 {
                &lfe_pcm
            } else {
                let i = channels.iter().position(|c| c.speaker == *s).expect("every speaker has a channel");
                &pcm[i]
            };
            for (i, v) in src.iter().enumerate() {
                out[i * nout + slot] = (v * OUTPUT_SCALE) as f32;
            }
        }
        info.layout = layout;
        info.sample_rate = out_rate;
        info.extensions = used;
        info.skipped = skipped;
        debug_assert!(!info.extensions.any() || !self.core_only);
        self.info = Some(info);
        Ok(Frame { samples: out, sample_rate: out_rate, channels: nout, layout })
    }
}

/// An embedded downmix to undo: `targets[i].0 -= targets[i].1 · source`,
/// after dividing the first `scale.1` channels by `scale.0` when given.
struct Downmix {
    source: usize,
    targets: Vec<(usize, f64)>,
    scale: Option<(f64, usize)>,
}
