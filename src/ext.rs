//! The core extensions of clause 6: XCh (6.4), XXCH (6.5), X96 (6.2) and
//! XBR (6.3). Each one's frame is located (by the caller) at its DWORD-
//! aligned sync word, inside the core frame or as a component of an
//! extension substream asset, and decoded here into subband samples that
//! join the core's before synthesis.

use super::Error;
use super::adpcm::{AdpcmFallback, PredictorState};
use super::bits::BitReader;
use super::core::{self, AudioCtx, ChannelBuf, CodingParams, MAX_SET, NSB};
use super::crc::crc16;
use super::huffman::{self, Codebook};
use super::tables;

/// XCh sync word (Table 6-17).
pub(crate) const SYNC_XCH: u32 = 0x5A5A_5A5A;
/// XXCH sync word (Table 6-21).
pub(crate) const SYNC_XXCH: u32 = 0x4700_4A03;
/// X96 sync word (Tables 6-1, 6-3).
pub(crate) const SYNC_X96: u32 = 0x1D95_F262;
/// XBR sync word (Table 6-12).
pub(crate) const SYNC_XBR: u32 = 0x655E_315E;

/// What the extensions take from the core frame they extend: its subframe
/// structure (they share it, 6.2.4.1, 6.3.4, 6.5.1), its flags and, for
/// XBR, its per-subframe `TMODE` and its `SHUFF`.
pub(crate) struct CoreFrameCtx {
    /// `nSSC` of each subframe.
    pub ssc: Vec<usize>,
    pub aspf: bool,
    pub cpf: bool,
    pub lossless_steps: bool,
}

impl CoreFrameCtx {
    fn step_table(&self) -> &'static [u32; 27] {
        if self.lossless_steps {
            &tables::STEP_SIZE_LOSSLESS_Q22
        } else {
            &tables::STEP_SIZE_LOSSY_Q22
        }
    }
}

/// Refuse a subframe that predicts without a code book, unless the caller
/// asked for the estimate. The message names the missing table.
pub(crate) fn check_prediction(predicted: usize, ctx: &AudioCtx, what: &str) -> Result<(), Error> {
    if predicted > 0 && ctx.adpcm_book.is_none() && ctx.fallback == AdpcmFallback::Refuse {
        return Err(Error::Unsupported(format!(
            "ADPCM prediction (PMODE = 1 in {predicted} subbands{what}): the D.10.1 prediction-coefficient \
             VQ codebook is not published in ETSI TS 102 114, so the residuals cannot be reconstructed \
             (supply it with Decoder::set_adpcm_codebook, or opt into AdpcmFallback::Estimate)"
        )));
    }
    Ok(())
}

/// The subframes of an XCh or XXCH channel set (Tables 6-19/6-20,
/// 6-24/6-25): side information, VQ subbands, subsubframes, joint coding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_set_subframes(
    r: &mut BitReader,
    p: &CodingParams,
    lower: &[&ChannelBuf],
    cc: &CoreFrameCtx,
    ctx: &AudioCtx,
    bufs: &mut [ChannelBuf],
    pred: &mut [PredictorState],
    hf_skipped: &mut bool,
    what: &str,
) -> Result<Vec<[[u32; NSB]; MAX_SET]>, Error> {
    let mut t0 = 0;
    let mut tmodes = Vec::with_capacity(cc.ssc.len());
    for &ssc in &cc.ssc {
        let si = core::parse_side_info(r, p, ssc)?;
        tmodes.push(si.tmode);
        if cc.cpf {
            let _sicrc = r.bits(16)?;
        }
        check_prediction(si.predicted(p), ctx, what)?;
        core::read_hf_vq(r, p, &si, t0, bufs, ctx.hf_book, hf_skipped)?;
        for ssf in 0..ssc {
            core::read_subsubframe(r, p, &si, ssf, t0 + 8 * ssf, ctx, bufs, pred)?;
        }
        core::finish_estimates(p, &si, t0, 8 * ssc, ctx, bufs, pred);
        core::apply_joint(p, &si, bufs, lower, t0, 8 * ssc);
        t0 += 8 * ssc;
    }
    Ok(tmodes)
}

/// The XCh frame header and audio header (Tables 6-17, 6-18), from the
/// sync word: `(XChFSIZE, channel count, coding params)`.
pub(crate) fn parse_xch_header(
    r: &mut BitReader,
    cpf: bool,
    lower_subs: &[usize],
) -> Result<(usize, CodingParams), Error> {
    if r.bits(32)? != SYNC_XCH {
        return Err(Error::Invalid("XCh sync word"));
    }
    let fsize = r.bits(10)? as usize;
    let _amode = r.bits(4)?;
    let n = r.bits(3)? as usize + 1;
    let p = core::parse_coding_params(r, n, lower_subs)?;
    if cpf {
        let _ahcrc = r.bits(16)?;
    }
    Ok((fsize, p))
}

/// Per channel of an XXCH set, `(target speaker mask bit, gain)` pairs.
pub(crate) type DownmixCoeffs = Vec<Vec<(u32, f64)>>;

/// One XXCH channel set's header (Table 6-23).
pub(crate) struct XxchSet {
    pub params: CodingParams,
    /// Speaker mask bits of this set's channels, in channel order.
    pub channel_bits: Vec<u32>,
    /// `Some((coefficients per channel as (target mask bit, gain), scale))`
    /// when the encoder embedded a downmix of this set into the lower ones.
    pub downmix: Option<(DownmixCoeffs, f64)>,
}

/// The XXCH frame header (Table 6-21): `(core speaker mask, sets as byte
/// ranges relative to the sync word, bits for masks, CRC flag)`.
pub(crate) struct XxchHeader {
    pub core_mask: u32,
    pub bits4mask: u32,
    pub crc_chset: bool,
    /// Each set's `(header + data)` byte range from the sync word.
    pub sets: Vec<(usize, usize)>,
}

pub(crate) fn parse_xxch_header(data: &[u8]) -> Result<XxchHeader, Error> {
    let mut r = BitReader::new(data);
    if r.bits(32)? != SYNC_XXCH {
        return Err(Error::Invalid("XXCH sync word"));
    }
    let header_size = r.bits(6)? as usize + 1;
    let crc_chset = r.flag()?;
    let bits4mask = r.bits(5)? + 1;
    let nsets = r.bits(2)? as usize + 1;
    let mut sizes = Vec::with_capacity(nsets);
    for _ in 0..nsets {
        sizes.push(r.bits(14)? as usize + 1);
    }
    let core_mask = r.bits(bits4mask)?;
    if header_size < 6 || header_size > data.len() {
        return Err(Error::Invalid("XXCH header size"));
    }
    // CRC over the header from nuHeaderSizeXXCh (byte 4) through the byte
    // alignment, then the stored CRC: zero remainder.
    if crc16(&data[4..header_size]) != 0 {
        return Err(Error::Invalid("XXCH header CRC"));
    }
    let mut sets = Vec::with_capacity(nsets);
    let mut off = header_size;
    for s in sizes {
        if off + s > data.len() {
            return Err(Error::Invalid("XXCH channel set past the end of its frame"));
        }
        sets.push((off, off + s));
        off += s;
    }
    Ok(XxchHeader {
        core_mask,
        bits4mask,
        crc_chset,
        sets,
    })
}

/// C.6-style coefficient code (6 bits): 0 → 0, else `DmixTable[(code − 1) << 2]`.
fn c6_gain(code: u32) -> Result<f64, Error> {
    if code == 0 {
        return Ok(0.0);
    }
    let idx = ((code - 1) << 2) as usize;
    tables::DMIX_TABLE
        .get(idx)
        .map(|v| *v as f64 / 32768.0)
        .ok_or(Error::Invalid("downmix coefficient code out of range"))
}

/// One XXCH channel set header (Table 6-23), at the start of `r`.
pub(crate) fn parse_xxch_set_header(
    r: &mut BitReader,
    set_bytes: &[u8],
    h: &XxchHeader,
    lower_subs: &[usize],
) -> Result<(XxchSet, usize), Error> {
    let start = r.position_bits();
    let header_size = r.bits(7)? as usize + 1;
    let n = r.bits(3)? as usize + 1;
    let layout_mask = r.bits(h.bits4mask.saturating_sub(6))? << 6;
    let mut downmix = None;
    if r.flag()? {
        let embedded = r.flag()?;
        let scale_code = r.bits(6)?;
        let mut maps = Vec::with_capacity(n);
        for _ in 0..n {
            maps.push(r.bits(h.bits4mask)?);
        }
        let mut coeffs = Vec::with_capacity(n);
        for map in maps {
            let mut c = Vec::new();
            for bit in 0..h.bits4mask {
                if (map >> bit) & 1 == 1 {
                    // 7-bit code: sign (1 = positive, as C.9), then a 6-bit
                    // C.6 table code.
                    let code = r.bits(7)?;
                    let sign = if code & 0x40 != 0 { 1.0 } else { -1.0 };
                    c.push((bit, sign * c6_gain(code & 0x3F)?));
                }
            }
            coeffs.push(c);
        }
        if embedded {
            let scale = c6_gain(scale_code)?;
            if scale <= 0.0 {
                return Err(Error::Invalid("XXCH downmix scale of zero"));
            }
            downmix = Some((coeffs, scale));
        }
    }
    let params = core::parse_coding_params(r, n, lower_subs)?;
    // Skip the reserved field and alignment to the declared header size.
    let used = (r.position_bits() - start).div_ceil(8);
    if used > header_size {
        return Err(Error::Invalid("XXCH channel set header overruns its size"));
    }
    if header_size > set_bytes.len() {
        return Err(Error::Invalid("XXCH channel set header past its set"));
    }
    if h.crc_chset && crc16(&set_bytes[..header_size]) != 0 {
        return Err(Error::Invalid("XXCH channel set header CRC"));
    }
    let channel_bits: Vec<u32> = (0..32).filter(|b| (layout_mask >> b) & 1 == 1).collect();
    if channel_bits.len() != n {
        return Err(Error::Invalid(
            "XXCH speaker mask disagrees with the channel count",
        ));
    }
    Ok((
        XxchSet {
            params,
            channel_bits,
            downmix,
        },
        header_size,
    ))
}

// ---------------------------------------------------------------------------
// X96
// ---------------------------------------------------------------------------

/// One X96 channel set header (Table 6-4).
pub(crate) struct X96Set {
    pub n: usize,
    pub sbs: usize,
    /// `anSBE96`: one past the last coded subband.
    pub sbe: [usize; MAX_SET],
    pub joinx: [usize; MAX_SET],
    pub shuff: [u32; MAX_SET],
    pub bhuff: [u32; MAX_SET],
    pub highres: bool,
    /// `SEL96[ch][k]` for the core quantiser `ABITS` = k + 1.
    pub sel: [[u32; 15]; MAX_SET],
}

pub(crate) fn parse_x96_set_header(
    r: &mut BitReader,
    n: usize,
    revno: u32,
    exss: bool,
) -> Result<X96Set, Error> {
    if n == 0 || n > MAX_SET {
        return Err(Error::Invalid("X96 channel count"));
    }
    let highres = r.flag()?;
    let sbs = if revno < 8 {
        let v = r.bits(5)? as usize;
        if v > 27 {
            return Err(Error::Invalid("X96 nSBS96 above 27"));
        }
        v
    } else {
        32
    };
    let mut s = X96Set {
        n,
        sbs,
        sbe: [0; MAX_SET],
        joinx: [0; MAX_SET],
        shuff: [0; MAX_SET],
        bhuff: [0; MAX_SET],
        highres,
        sel: [[0; 15]; MAX_SET],
    };
    for ch in 0..n {
        s.sbe[ch] = r.bits(6)? as usize + 1;
        if s.sbe[ch] < s.sbs || (!exss && s.sbe[ch] < 32) {
            return Err(Error::Invalid("X96 SBE96 below the first coded subband"));
        }
    }
    for ch in 0..n {
        s.joinx[ch] = r.bits(3)? as usize;
        if s.joinx[ch] > ch {
            return Err(Error::Invalid(
                "JOINX96 source channel is not a lower channel",
            ));
        }
    }
    for ch in 0..n {
        s.shuff[ch] = r.bits(3)?;
        if s.shuff[ch] > 5 {
            return Err(Error::Invalid("SHUFF96 above 5"));
        }
    }
    for ch in 0..n {
        s.bhuff[ch] = r.bits(3)?;
    }
    // SEL96: 1 bit for 3 levels, 2 bits for 5…13 levels, 3 bits for 17
    // levels and, with HIGHRESFLAG96K, for 25…129 levels (Table 6-8).
    // Table 6-4 prints the high-resolution loop from n = 5 again; the
    // 17-level selector is read once (see the crate docs).
    for ch in 0..n {
        s.sel[ch][0] = r.bits(1)?;
    }
    for k in 1..5 {
        for ch in 0..n {
            s.sel[ch][k] = r.bits(2)?;
        }
    }
    let last = if highres { 10 } else { 6 };
    for k in 5..last {
        for ch in 0..n {
            s.sel[ch][k] = r.bits(3)?;
        }
    }
    Ok(s)
}

fn x96_abits_book(highres: bool, bhuff: u32) -> Option<&'static Codebook> {
    if bhuff == 7 {
        None
    } else if highres {
        Some(huffman::book_33(bhuff))
    } else {
        Some(huffman::book_17(bhuff))
    }
}

/// A small deterministic generator for the X96 noise fill (`ABITS96` = 0,
/// Table 6-10: uniform in [−0.5, 0.5] × the scale factor).
pub(crate) struct Noise(u32);

impl Default for Noise {
    fn default() -> Self {
        Noise(0x1234_5678)
    }
}

impl Noise {
    fn next(&mut self) -> f64 {
        // xorshift32
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as f64 / u32::MAX as f64 - 0.5
    }
}

/// The subframes of one X96 channel set (Tables 6-9, 6-10): residual
/// subband samples for 64 bands into `bufs`, which the caller then adds the
/// core's subband samples to.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_x96_subframes(
    r: &mut BitReader,
    s: &X96Set,
    cc: &CoreFrameCtx,
    ctx: &AudioCtx,
    bufs: &mut [ChannelBuf],
    pred: &mut [PredictorState],
    noise: &mut Noise,
    hf_skipped: &mut bool,
) -> Result<(), Error> {
    const NB: usize = 64;
    let step_table = cc.step_table();
    let mut t0 = 0;
    for &ssc in &cc.ssc {
        let nsamp = 8 * ssc;
        let mut pmode = [[false; NB]; MAX_SET];
        let mut pvq = [[0u16; NB]; MAX_SET];
        let mut abits = [[0u32; NB]; MAX_SET];
        let mut scales = [[0.0f64; NB]; MAX_SET];
        for ch in 0..s.n {
            for sb in s.sbs..s.sbe[ch] {
                pmode[ch][sb] = r.flag()?;
            }
        }
        let mut predicted = 0;
        for ch in 0..s.n {
            for sb in s.sbs..s.sbe[ch] {
                if pmode[ch][sb] {
                    pvq[ch][sb] = r.bits(12)? as u16;
                    predicted += 1;
                }
            }
        }
        check_prediction(predicted, ctx, " of X96")?;
        for ch in 0..s.n {
            let book = x96_abits_book(s.highres, s.bhuff[ch]);
            let max = if s.highres { 15 } else { 7 };
            let mut acc = 0i32;
            for sb in s.sbs..s.sbe[ch] {
                let v = match book {
                    Some(b) => {
                        acc += b.decode(r)?;
                        acc
                    }
                    None => r.bits(if s.highres { 4 } else { 3 })? as i32,
                };
                if !(0..=max).contains(&v) {
                    return Err(Error::Invalid("ABITS96 out of range"));
                }
                abits[ch][sb] = v as u32;
            }
        }
        for ch in 0..s.n {
            let shuff = s.shuff[ch];
            let mut acc = 0i32;
            for sb in s.sbs..s.sbe[ch] {
                let idx = if shuff == 5 {
                    r.bits(6)? as i32
                } else {
                    acc += huffman::scale_book(shuff)
                        .expect("SHUFF96 < 5 has a book")
                        .decode(r)?;
                    acc
                };
                scales[ch][sb] = usize::try_from(idx)
                    .ok()
                    .and_then(|i| tables::SCALE_RMS_6BIT.get(i).copied())
                    .filter(|v| *v > 0)
                    .ok_or(Error::Invalid("SCALES96 index outside the 6-bit table"))?
                    as f64;
            }
        }
        let mut join_shuff = [0u32; MAX_SET];
        for ch in 0..s.n {
            if s.joinx[ch] > 0 {
                join_shuff[ch] = r.bits(3)?;
            }
        }
        let mut join_scales = [[0.0f64; NB]; MAX_SET];
        for ch in 0..s.n {
            if s.joinx[ch] > 0 {
                let src = s.joinx[ch] - 1;
                let book = huffman::scale_book(join_shuff[ch]);
                for sb in s.sbe[ch]..s.sbe[src] {
                    let raw = match join_shuff[ch] {
                        5 => r.bits(6)? as i32,
                        6 => r.bits(7)? as i32,
                        _ => book
                            .ok_or(Error::Invalid("JOIN_SHUFF96 without a code book"))?
                            .decode(r)?,
                    };
                    join_scales[ch][sb] = usize::try_from(raw + 64)
                        .ok()
                        .and_then(|i| tables::JOINT_INTENSITY_SCALE.get(i).copied())
                        .ok_or(Error::Invalid("joint intensity scale index out of range"))?
                        as f64;
                }
            }
        }
        if cc.cpf {
            let _sicrc96 = r.bits(16)?;
        }
        // HFREQ96 and the noise fill.
        for ch in 0..s.n {
            for sb in s.sbs..s.sbe[ch] {
                let out = &mut bufs[ch].s[sb][t0..t0 + nsamp];
                match abits[ch][sb] {
                    0 => {
                        for o in out.iter_mut() {
                            *o = noise.next() * scales[ch][sb];
                        }
                    }
                    1 => {
                        for chunk in 0..nsamp.div_ceil(16) {
                            let index = r.bits(10)? as usize;
                            let len = (nsamp - chunk * 16).min(16);
                            match ctx.hf_book {
                                Some(b) => {
                                    for m in 0..len {
                                        out[chunk * 16 + m] = b.element(index, m) * scales[ch][sb];
                                    }
                                }
                                None => {
                                    *hf_skipped = true;
                                    out[chunk * 16..chunk * 16 + len].fill(0.0);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Audio codes per subsubframe.
        let mut q = [0i32; 8];
        for ssf in 0..ssc {
            for ch in 0..s.n {
                for sb in s.sbs..s.sbe[ch] {
                    let a = abits[ch][sb];
                    if a < 2 {
                        continue;
                    }
                    let core_abits = a - 1;
                    let sel = s.sel[ch].get(core_abits as usize - 1).copied().unwrap_or(0);
                    let coding =
                        huffman::sample_coding(core_abits, if core_abits <= 10 { sel } else { 0 })?;
                    core::read_indices(r, coding, &mut q)?;
                    let step = step_table[core_abits as usize] as f64 / (1u32 << 22) as f64;
                    let scale = step * scales[ch][sb];
                    let t = t0 + 8 * ssf;
                    for (o, v) in bufs[ch].s[sb][t..t + 8].iter_mut().zip(&q) {
                        *o = scale * *v as f64;
                    }
                }
            }
            if (ssf == ssc - 1 || cc.aspf) && r.bits(16)? != 0xFFFF {
                return Err(Error::Invalid("DSYNC missing in X96"));
            }
        }
        // Inverse ADPCM over the whole subframe, every predicted subband.
        for ch in 0..s.n {
            for sb in s.sbs..s.sbe[ch] {
                if pmode[ch][sb] {
                    match ctx.adpcm_book {
                        Some(b) => {
                            let coeffs = b.coefficients(pvq[ch][sb] as usize);
                            for ssf in 0..ssc {
                                let t = t0 + 8 * ssf;
                                pred[ch].inverse(sb, &coeffs, &mut bufs[ch].s[sb][..t + 8], t);
                            }
                        }
                        None => {
                            pred[ch].estimate_subframe(sb, &mut bufs[ch].s[sb][..t0 + nsamp], t0);
                        }
                    }
                }
            }
        }
        // Joint intensity, and clear what is not coded.
        for ch in 0..s.n {
            let end = if s.joinx[ch] > 0 {
                let src = s.joinx[ch] - 1;
                for sb in s.sbe[ch]..s.sbe[src] {
                    for t in t0..t0 + nsamp {
                        bufs[ch].s[sb][t] = join_scales[ch][sb] * bufs[src].s[sb][t];
                    }
                }
                s.sbe[src].max(s.sbe[ch])
            } else {
                s.sbe[ch]
            };
            for sb in (0..s.sbs).chain(end..NB) {
                bufs[ch].s[sb][t0..t0 + nsamp].fill(0.0);
            }
        }
        t0 += nsamp;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// XBR
// ---------------------------------------------------------------------------

/// One channel an XBR channel set extends: its subband samples, and the
/// `SHUFF` and per-subframe `TMODE`s it was coded with.
pub(crate) struct XbrTarget<'a> {
    pub buf: &'a mut ChannelBuf,
    pub shuff: u32,
    pub tmode: &'a [[u32; NSB]],
}

/// The XBR frame (Tables 6-12 … 6-16), adding its residuals into the
/// channel sets' subband samples: `targets[set]` are the set's channels.
pub(crate) fn decode_xbr(
    data: &[u8],
    cc: &CoreFrameCtx,
    targets: &mut [Vec<XbrTarget>],
) -> Result<(), Error> {
    let mut r = BitReader::new(data);
    if r.bits(32)? != SYNC_XBR {
        return Err(Error::Invalid("XBR sync word"));
    }
    let header_size = r.bits(6)? as usize + 1;
    let nsets = r.bits(2)? as usize + 1;
    let mut sizes = Vec::with_capacity(nsets);
    for _ in 0..nsets {
        sizes.push(r.bits(14)? as usize + 1);
    }
    let tmode_flag = r.flag()?;
    let mut active: Vec<Vec<usize>> = Vec::with_capacity(nsets);
    for _ in 0..nsets {
        let nch = r.bits(3)? as usize + 1;
        let bits = r.bits(2)? + 5;
        let mut a = Vec::with_capacity(nch);
        for _ in 0..nch {
            a.push(r.bits(bits)? as usize + 1);
        }
        active.push(a);
    }
    if header_size < 6 || header_size > data.len() || crc16(&data[4..header_size]) != 0 {
        return Err(Error::Invalid("XBR header CRC"));
    }
    let step_table = &tables::STEP_SIZE_LOSSLESS_Q22;
    let mut off = header_size;
    for (set, size) in sizes.iter().enumerate() {
        let end = off + size;
        if end > data.len() {
            return Err(Error::Invalid("XBR channel set past the end of its frame"));
        }
        let Some(target) = targets.get_mut(set) else {
            break; // a set for channels not decoded here
        };
        let bands = &active[set];
        if bands.len() > target.len() {
            return Err(Error::Invalid(
                "XBR channel set larger than the channels it extends",
            ));
        }
        let mut r = BitReader::new(&data[off..end]);
        let mut t0 = 0;
        for (sf, &ssc) in cc.ssc.iter().enumerate() {
            let nch = bands.len();
            let mut abits_bits = [0u32; MAX_SET];
            for b in abits_bits.iter_mut().take(nch) {
                *b = r.bits(2)? + 2;
            }
            let mut abits: Vec<Vec<u32>> = Vec::with_capacity(nch);
            for ch in 0..nch {
                let mut a = Vec::with_capacity(bands[ch]);
                for _ in 0..bands[ch] {
                    a.push(r.bits(abits_bits[ch])?);
                }
                abits.push(a);
            }
            let mut scale_bits = [0u32; MAX_SET];
            for b in scale_bits.iter_mut().take(nch) {
                *b = r.bits(3)?;
                if *b < 1 {
                    return Err(Error::Invalid("XBR scale index width of zero"));
                }
            }
            let mut scales: Vec<Vec<[f64; 2]>> = Vec::with_capacity(nch);
            for ch in 0..nch {
                let shuff = target[ch].shuff;
                let mut sc = Vec::with_capacity(bands[ch]);
                for sb in 0..bands[ch] {
                    let mut pair = [0.0; 2];
                    if abits[ch][sb] > 0 {
                        let look = |i: usize| -> Result<f64, Error> {
                            let v = if shuff == 6 {
                                tables::SCALE_RMS_7BIT.get(i).copied()
                            } else {
                                tables::SCALE_RMS_6BIT.get(i).copied()
                            };
                            v.map(|v| v as f64)
                                .ok_or(Error::Invalid("XBR scale index outside the table"))
                        };
                        pair[0] = look(r.bits(scale_bits[ch])? as usize)?;
                        let tm = target[ch].tmode.get(sf).map_or(0, |t| t[sb.min(NSB - 1)]);
                        if tmode_flag && tm > 0 {
                            pair[1] = look(r.bits(scale_bits[ch])? as usize)?;
                        }
                    }
                    sc.push(pair);
                }
                scales.push(sc);
            }
            for ssf in 0..ssc {
                for ch in 0..nch {
                    let blocks = target[ch].buf.s.first().map_or(0, |v| v.len());
                    for sb in 0..bands[ch] {
                        let a = abits[ch][sb] as usize;
                        let mut q = [0i32; 8];
                        if a > 7 {
                            for v in q.iter_mut() {
                                *v = r.sbits(a as u32 - 3)?;
                            }
                        } else if a > 0 {
                            let levels = [3u32, 5, 7, 9, 13, 17, 25][a - 1];
                            let bits = tables::BLOCK_CODE_BITS[a - 1].1 as u32;
                            let mut block = [0i32; 4];
                            for half in 0..2 {
                                huffman::decode_block(r.bits(bits)?, levels, &mut block)?;
                                q[half * 4..half * 4 + 4].copy_from_slice(&block);
                            }
                        } else {
                            continue;
                        }
                        let step = *step_table
                            .get(a)
                            .ok_or(Error::Invalid("XBR ABITS above 26"))?
                            as f64
                            / (1u32 << 22) as f64;
                        let tm = if tmode_flag {
                            target[ch].tmode.get(sf).map_or(0, |t| t[sb.min(NSB - 1)])
                        } else {
                            0
                        };
                        let sf_ = if tm == 0 || ssf < tm as usize {
                            scales[ch][sb][0]
                        } else {
                            scales[ch][sb][1]
                        };
                        let t = t0 + 8 * ssf;
                        if sb < target[ch].buf.s.len() && t + 8 <= blocks {
                            for (m, v) in q.iter().enumerate() {
                                target[ch].buf.s[sb][t + m] += step * sf_ * *v as f64;
                            }
                        }
                    }
                }
                if (ssf == ssc - 1 || cc.aspf) && r.bits(16)? != 0xFFFF {
                    return Err(Error::Invalid("DSYNC missing in XBR"));
                }
            }
            t0 += 8 * ssc;
        }
        off = end;
    }
    Ok(())
}
