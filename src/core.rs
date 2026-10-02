//! The subband coding shared by the core (clause 5) and the XCh and XXCH
//! channel extensions (6.4, 6.5): the per-channel coding header from `SUBS`
//! to `ADJ` (Tables 5-21, 6-18, 6-23), the subframe side information
//! (Tables 5-28, 6-19, 6-24) and the audio data arrays (Tables 5-29, 6-20,
//! 6-25). The three clauses print the same syntax three times; it is written
//! once here, for a "coding set" of up to eight channels.
//!
//! Subband samples land in a [`SetBuffers`]: per channel, per subband, one
//! sample per 32-PCM-sample block of the frame. Inverse ADPCM (Annex C.3.3)
//! runs per subsubframe as the samples are dequantised, against the
//! subband's own reconstructed history ([`crate::adpcm`]).

use super::adpcm::{AdpcmFallback, PredictorState};
use super::bits::BitReader;
use super::huffman::{self, SampleCoding};
use super::tables;
use super::vq::{AdpcmCodebook, HfVqCodebook};
use super::Error;

/// Subbands of the core filter bank.
pub(crate) const NSB: usize = 32;
/// Channels one coding set can carry (`PCHS` is 3 bits).
pub(crate) const MAX_SET: usize = 8;

/// Table 5-27: scale factor adjustment values by `ADJ` index.
const ADJ_TABLE: [f64; 4] = [1.0, 1.125, 1.25, 1.4375];

/// The per-channel coding header, `SUBS` through `ADJ`.
#[derive(Clone)]
pub(crate) struct CodingParams {
    pub n: usize,
    pub subs: [usize; MAX_SET],
    pub vqsub: [usize; MAX_SET],
    pub joinx: [usize; MAX_SET],
    pub thuff: [u32; MAX_SET],
    pub shuff: [u32; MAX_SET],
    pub bhuff: [u32; MAX_SET],
    /// `SEL[ch][n]` for `ABITS` = n+1, n < 10 (Table 5-26).
    pub sel: [[u32; 10]; MAX_SET],
    /// `arADJ[ch][n]`, 1.0 unless transmitted.
    pub adj: [[f64; 10]; MAX_SET],
}

/// Read `SUBS` … `ADJ` for `n` channels.
pub(crate) fn parse_coding_params(r: &mut BitReader, n: usize) -> Result<CodingParams, Error> {
    if n == 0 || n > MAX_SET {
        return Err(Error::Invalid("channel count of a coding set out of range"));
    }
    let mut p = CodingParams {
        n,
        subs: [0; MAX_SET],
        vqsub: [0; MAX_SET],
        joinx: [0; MAX_SET],
        thuff: [0; MAX_SET],
        shuff: [0; MAX_SET],
        bhuff: [0; MAX_SET],
        sel: [[0; 10]; MAX_SET],
        adj: [[1.0; 10]; MAX_SET],
    };
    for ch in 0..n {
        p.subs[ch] = r.bits(5)? as usize + 2;
        if p.subs[ch] > NSB {
            return Err(Error::Invalid("SUBS above 32 active subbands"));
        }
    }
    for ch in 0..n {
        p.vqsub[ch] = r.bits(5)? as usize + 1;
        if p.vqsub[ch] > p.subs[ch] {
            return Err(Error::Invalid("VQSUB above SUBS"));
        }
    }
    for ch in 0..n {
        p.joinx[ch] = r.bits(3)? as usize;
        if p.joinx[ch] > ch {
            return Err(Error::Invalid("JOINX source channel is not a lower channel"));
        }
    }
    for ch in 0..n {
        p.thuff[ch] = r.bits(2)?;
    }
    for ch in 0..n {
        p.shuff[ch] = r.bits(3)?;
        if p.shuff[ch] == 7 {
            return Err(Error::Invalid("SHUFF = 7"));
        }
    }
    for ch in 0..n {
        p.bhuff[ch] = r.bits(3)?;
        if p.bhuff[ch] == 7 {
            return Err(Error::Invalid("BHUFF = 7"));
        }
    }
    for ch in 0..n {
        p.sel[ch][0] = r.bits(1)?;
    }
    for k in 1..5 {
        for ch in 0..n {
            p.sel[ch][k] = r.bits(2)?;
        }
    }
    for k in 5..10 {
        for ch in 0..n {
            p.sel[ch][k] = r.bits(3)?;
        }
    }
    // ADJ is transmitted wherever SEL picked a Huffman book.
    for ch in 0..n {
        if p.sel[ch][0] == 0 {
            p.adj[ch][0] = ADJ_TABLE[r.bits(2)? as usize];
        }
    }
    for k in 1..5 {
        for ch in 0..n {
            if p.sel[ch][k] < 3 {
                p.adj[ch][k] = ADJ_TABLE[r.bits(2)? as usize];
            }
        }
    }
    for k in 5..10 {
        for ch in 0..n {
            if p.sel[ch][k] < 7 {
                p.adj[ch][k] = ADJ_TABLE[r.bits(2)? as usize];
            }
        }
    }
    Ok(p)
}

/// One subframe's side information, `PMODE` through `JOIN_SCALES`.
pub(crate) struct SideInfo {
    pub ssc: usize,
    pub pmode: [[bool; NSB]; MAX_SET],
    pub pvq: [[u16; NSB]; MAX_SET],
    pub abits: [[u32; NSB]; MAX_SET],
    pub tmode: [[u32; NSB]; MAX_SET],
    pub scales: [[[f64; 2]; NSB]; MAX_SET],
    pub join_scales: [[f64; NSB]; MAX_SET],
}

impl SideInfo {
    /// Subbands with `PMODE` = 1 below `VQSUB` (the ones whose prediction
    /// is inverted; `PMODE` above `VQSUB` has no effect, Table 5-29).
    pub fn predicted(&self, p: &CodingParams) -> usize {
        (0..p.n).map(|ch| (0..p.vqsub[ch]).filter(|&sb| self.pmode[ch][sb]).count()).sum()
    }
}

fn read_scale_index(r: &mut BitReader, shuff: u32, acc: &mut i32) -> Result<usize, Error> {
    let idx = match shuff {
        5 => r.bits(6)? as i32,
        6 => r.bits(7)? as i32,
        _ => {
            // Huffman: the difference from the previous index.
            *acc += huffman::scale_book(shuff).expect("SHUFF < 5 has a book").decode(r)?;
            *acc
        }
    };
    if shuff >= 5 {
        *acc = idx;
    }
    if idx < 0 {
        return Err(Error::Invalid("negative scale factor index"));
    }
    Ok(idx as usize)
}

fn scale_lookup(shuff: u32, idx: usize) -> Result<f64, Error> {
    let v = if shuff == 6 {
        tables::SCALE_RMS_7BIT.get(idx).copied()
    } else {
        tables::SCALE_RMS_6BIT.get(idx).copied()
    };
    match v {
        Some(v) if v > 0 => Ok(v as f64),
        _ => Err(Error::Invalid("scale factor index outside the quantisation table")),
    }
}

/// Read one subframe's side information (Table 5-28 from `PMODE` to
/// `JOIN_SCALES`; the same in Tables 6-19 and 6-24). `ssc` is `nSSC`.
pub(crate) fn parse_side_info(r: &mut BitReader, p: &CodingParams, ssc: usize) -> Result<SideInfo, Error> {
    let mut si = SideInfo {
        ssc,
        pmode: [[false; NSB]; MAX_SET],
        pvq: [[0; NSB]; MAX_SET],
        abits: [[0; NSB]; MAX_SET],
        tmode: [[0; NSB]; MAX_SET],
        scales: [[[0.0; 2]; NSB]; MAX_SET],
        join_scales: [[0.0; NSB]; MAX_SET],
    };
    for ch in 0..p.n {
        for sb in 0..p.subs[ch] {
            si.pmode[ch][sb] = r.flag()?;
        }
    }
    for ch in 0..p.n {
        for sb in 0..p.subs[ch] {
            if si.pmode[ch][sb] {
                si.pvq[ch][sb] = r.bits(12)? as u16;
            }
        }
    }
    // ABITS (bit allocation) per non-VQ subband.
    for ch in 0..p.n {
        let book = huffman::abits_book(p.bhuff[ch]);
        for sb in 0..p.vqsub[ch] {
            si.abits[ch][sb] = match p.bhuff[ch] {
                5 => r.bits(4)?,
                6 => r.bits(5)?,
                _ => {
                    let v = book.expect("BHUFF < 5 has a book").decode(r)?;
                    u32::try_from(v).map_err(|_| Error::Invalid("negative ABITS"))?
                }
            };
            if si.abits[ch][sb] > 26 {
                return Err(Error::Invalid("ABITS above 26"));
            }
        }
    }
    // TMODE, only with more than one subsubframe and only where bits are allocated.
    if ssc > 1 {
        for ch in 0..p.n {
            let book = huffman::tmode_book(p.thuff[ch]);
            for sb in 0..p.vqsub[ch] {
                if si.abits[ch][sb] > 0 {
                    si.tmode[ch][sb] = book.decode(r)? as u32;
                }
            }
        }
    }
    // SCALES: one per allocated subband, two if it has a transient; then one
    // per high-frequency VQ subband. Huffman-coded indices are differences
    // accumulated across the channel.
    for ch in 0..p.n {
        let shuff = p.shuff[ch];
        let mut acc = 0i32;
        for sb in 0..p.vqsub[ch] {
            if si.abits[ch][sb] > 0 {
                let idx = read_scale_index(r, shuff, &mut acc)?;
                si.scales[ch][sb][0] = scale_lookup(shuff, idx)?;
                if si.tmode[ch][sb] > 0 {
                    let idx = read_scale_index(r, shuff, &mut acc)?;
                    si.scales[ch][sb][1] = scale_lookup(shuff, idx)?;
                }
            }
        }
        for sb in p.vqsub[ch]..p.subs[ch] {
            let idx = read_scale_index(r, shuff, &mut acc)?;
            si.scales[ch][sb][0] = scale_lookup(shuff, idx)?;
        }
    }
    // Joint intensity coding: a code book select, then one scale per subband
    // copied from the source channel.
    let mut join_shuff = [0u32; MAX_SET];
    for ch in 0..p.n {
        if p.joinx[ch] > 0 {
            join_shuff[ch] = r.bits(3)?;
            if join_shuff[ch] == 7 {
                return Err(Error::Invalid("JOIN_SHUFF = 7"));
            }
        }
    }
    for ch in 0..p.n {
        if p.joinx[ch] > 0 {
            let src = p.joinx[ch] - 1;
            si.join_scales[ch] = read_join_scales(r, join_shuff[ch], p.subs[ch], p.subs[src])?;
        }
    }
    Ok(si)
}

/// `JOIN_SCALES` for subbands `from..to`: the `JOIN_SHUFF` code, biased by
/// 64, into D.3.
pub(crate) fn read_join_scales(r: &mut BitReader, shuff: u32, from: usize, to: usize) -> Result<[f64; NSB], Error> {
    let mut out = [0.0; NSB];
    let book = huffman::scale_book(shuff);
    for v in out.iter_mut().take(to).skip(from) {
        let raw = match shuff {
            5 => r.bits(6)? as i32,
            6 => r.bits(7)? as i32,
            _ => book.ok_or(Error::Invalid("JOIN_SHUFF without a code book"))?.decode(r)?,
        };
        let scale = usize::try_from(raw + 64)
            .ok()
            .and_then(|i| tables::JOINT_INTENSITY_SCALE.get(i).copied())
            .ok_or(Error::Invalid("joint intensity scale index out of range"))?;
        *v = scale as f64;
    }
    Ok(out)
}

/// Read the eight quantisation indices of one subband subsubframe.
pub(crate) fn read_indices(r: &mut BitReader, coding: SampleCoding, q: &mut [i32; 8]) -> Result<(), Error> {
    match coding {
        SampleCoding::None => q.fill(0),
        SampleCoding::Huffman(book) => {
            for v in q.iter_mut() {
                *v = book.decode(r)?;
            }
        }
        SampleCoding::Raw { bits } => {
            for v in q.iter_mut() {
                *v = r.sbits(bits)?;
            }
        }
        SampleCoding::Block { levels, bits } => {
            let mut block = [0i32; 4];
            for half in 0..2 {
                let code = r.bits(bits)?;
                huffman::decode_block(code, levels, &mut block)?;
                q[half * 4..half * 4 + 4].copy_from_slice(&block);
            }
        }
    }
    Ok(())
}

/// Per channel, per subband, the frame's subband samples: `s[sb][t]` for
/// block `t` (one sample per 32 core PCM samples).
pub(crate) struct ChannelBuf {
    pub s: Vec<Vec<f64>>,
}

impl ChannelBuf {
    pub fn new(bands: usize, blocks: usize) -> Self {
        Self { s: vec![vec![0.0; blocks]; bands] }
    }
}

/// What the audio-array reader needs besides the bit stream.
pub(crate) struct AudioCtx<'a> {
    pub step_table: &'a [u32; 27],
    pub aspf: bool,
    pub adpcm_book: Option<&'a AdpcmCodebook>,
    pub hf_book: Option<&'a HfVqCodebook>,
    pub fallback: AdpcmFallback,
}

/// The high-frequency VQ indices of one subframe (Table 5-29, first loop),
/// for the subframe starting at block `t0`. With the D.10.2 book the
/// samples are reconstructed (`SCALES · element`); without it they stay
/// zero and `*skipped` is set.
pub(crate) fn read_hf_vq(
    r: &mut BitReader,
    p: &CodingParams,
    si: &SideInfo,
    t0: usize,
    bufs: &mut [ChannelBuf],
    book: Option<&HfVqCodebook>,
    skipped: &mut bool,
) -> Result<(), Error> {
    for ch in 0..p.n {
        for sb in p.vqsub[ch]..p.subs[ch] {
            let index = r.bits(10)? as usize;
            match book {
                Some(b) => {
                    let scale = si.scales[ch][sb][0];
                    for m in 0..8 * si.ssc {
                        bufs[ch].s[sb][t0 + m] = scale * b.element(index, m);
                    }
                }
                None => *skipped = true,
            }
        }
    }
    Ok(())
}

/// One subsubframe of audio data for a coding set (Table 5-29 audio loop):
/// indices, dequantisation, inverse ADPCM, then the `DSYNC` check. `t` is
/// the first block of this subsubframe within the frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_subsubframe(
    r: &mut BitReader,
    p: &CodingParams,
    si: &SideInfo,
    ssf: usize,
    t: usize,
    ctx: &AudioCtx,
    bufs: &mut [ChannelBuf],
    pred: &mut [PredictorState],
) -> Result<(), Error> {
    let mut q = [0i32; 8];
    for ch in 0..p.n {
        for sb in 0..p.vqsub[ch] {
            let abits = si.abits[ch][sb];
            let sel = if (1..=10).contains(&abits) { p.sel[ch][abits as usize - 1] } else { 0 };
            let coding = huffman::sample_coding(abits, sel)?;
            read_indices(r, coding, &mut q)?;
            let out = &mut bufs[ch].s[sb][t..t + 8];
            if abits == 0 {
                out.fill(0.0);
            } else {
                let step = ctx.step_table[abits as usize] as f64 / (1u32 << 22) as f64;
                let tm = if si.tmode[ch][sb] == 0 { si.ssc } else { si.tmode[ch][sb] as usize };
                let sf = if ssf < tm { si.scales[ch][sb][0] } else { si.scales[ch][sb][1] };
                let mut scale = step * sf;
                if let SampleCoding::Huffman(_) = coding {
                    scale *= p.adj[ch][abits as usize - 1];
                }
                for (o, v) in out.iter_mut().zip(&q) {
                    *o = scale * *v as f64;
                }
            }
            if si.pmode[ch][sb] {
                let coeffs = match ctx.adpcm_book {
                    Some(b) => b.coefficients(si.pvq[ch][sb] as usize),
                    None => match ctx.fallback {
                        AdpcmFallback::Estimate => pred[ch].estimate(sb, &bufs[ch].s[sb][..t]),
                        AdpcmFallback::Refuse => {
                            unreachable!("a refused prediction is caught before the audio is read")
                        }
                    },
                };
                pred[ch].inverse(sb, &coeffs, &mut bufs[ch].s[sb][..t + 8], t);
            }
        }
    }
    if (ssf == si.ssc - 1 || ctx.aspf) && r.bits(16)? != 0xFFFF {
        return Err(Error::Invalid("DSYNC missing at the end of a subsubframe"));
    }
    Ok(())
}

/// Joint intensity coding (Annex C.3.4) over blocks `t0..t0 + n`.
pub(crate) fn apply_joint(p: &CodingParams, si: &SideInfo, bufs: &mut [ChannelBuf], t0: usize, n: usize) {
    for ch in 0..p.n {
        if p.joinx[ch] > 0 {
            let src = p.joinx[ch] - 1;
            for sb in p.subs[ch]..p.subs[src] {
                for s in t0..t0 + n {
                    bufs[ch].s[sb][s] = si.join_scales[ch][sb] * bufs[src].s[sb][s];
                }
            }
        }
    }
}
