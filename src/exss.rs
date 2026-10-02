//! The DTS-HD extension substream (clause 7): its header (Table 7-2) and
//! audio asset descriptors (Tables 7-5 … 7-7), parsed far enough to find
//! every coding component of every asset and what the asset is.

use std::ops::Range;

use super::bits::BitReader;
use super::crc::crc16;
use super::Error;

/// Extension substream sync word (Table 7-1).
pub(crate) const SYNC_EXSS: u32 = 0x6458_2025;
/// A core frame inside an extension substream (Table 7-1).
pub(crate) const SYNC_EXSS_CORE: u32 = 0x02B0_9261;

/// `nuCoreExtensionMask` bits (Table 7-15).
pub(crate) mod mask {
    pub const EXSS_CORE: u32 = 0x010;
    pub const EXSS_XBR: u32 = 0x020;
    pub const EXSS_XXCH: u32 = 0x040;
    pub const EXSS_X96: u32 = 0x080;
    pub const EXSS_LBR: u32 = 0x100;
    pub const EXSS_XLL: u32 = 0x200;
}

/// One audio asset of an extension substream frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct Asset {
    #[allow(dead_code)]
    pub index: u32,
    /// `nuCodingMode` (Table 7-14): 0 core + extensions, 1 lossless only,
    /// 2 LBR, 3 auxiliary.
    pub coding_mode: u32,
    /// `nuCoreExtensionMask` for coding mode 0; the implied component for
    /// the others.
    pub ext_mask: u32,
    /// Each component's bytes, relative to the extension substream frame,
    /// keyed by its `nuCoreExtensionMask` bit.
    pub components: Vec<(u32, Range<usize>)>,
    /// `nuSpkrActivityMask` (Table 7-10), when given.
    pub spkr_mask: Option<u32>,
    pub total_channels: u32,
    /// `nuMaxSampleRate` in Hz (Table 7-9), when the static fields are present.
    pub max_sample_rate: Option<u32>,
    pub bit_resolution: Option<u32>,
    /// `nuDTSHDStreamID`, present with XLL.
    pub stream_id: Option<u32>,
}

impl Asset {
    pub fn component(&self, bit: u32) -> Option<Range<usize>> {
        self.components.iter().find(|(b, _)| *b == bit).map(|(_, r)| r.clone())
    }
}

/// A parsed extension substream frame.
#[derive(Clone, Debug)]
#[allow(dead_code)] // diagnostics
pub(crate) struct ExssFrame {
    pub index: u32,
    /// `nuExtSSFsize`: the whole frame, header included.
    pub size: usize,
    /// `nuExSSFrameDurationCode` in reference clock cycles, when present.
    pub duration: Option<u32>,
    pub assets: Vec<Asset>,
    /// `nuBits4ExSSFsize`.
    pub bits4fsize: u32,
}

/// Table 7-9.
const SAMPLE_RATES_7_9: [u32; 16] = [
    8_000, 16_000, 32_000, 64_000, 128_000, 22_050, 44_100, 88_200, 176_400, 352_800, 12_000, 24_000, 48_000,
    96_000, 192_000, 384_000,
];

/// `NumSpkrTableLookUp`: channels of a Table 7-10 mask (pairs count two).
pub(crate) fn num_speakers(mask: u32) -> u32 {
    const PAIRS: u32 = 0x0002 | 0x0004 | 0x0020 | 0x0040 | 0x0200 | 0x0400 | 0x0800 | 0x2000 | 0x8000;
    (0..16)
        .filter(|b| (mask >> b) & 1 == 1)
        .map(|b| if (PAIRS >> b) & 1 == 1 { 2 } else { 1 })
        .sum()
}

/// Read the size of the extension substream frame at the start of `buf`
/// without parsing the rest (for packet splitting).
pub(crate) fn frame_size(buf: &[u8]) -> Result<usize, Error> {
    let mut r = BitReader::new(buf);
    if r.bits(32)? != SYNC_EXSS {
        return Err(Error::NoSync);
    }
    let _user = r.bits(8)?;
    let _index = r.bits(2)?;
    let long = r.flag()?;
    let (bh, bf) = if long { (12, 20) } else { (8, 16) };
    let _header = r.bits(bh)?;
    Ok(r.bits(bf)? as usize + 1)
}

/// Parse the extension substream frame at the start of `buf` (which must
/// hold all of it).
pub(crate) fn parse(buf: &[u8]) -> Result<ExssFrame, Error> {
    let mut r = BitReader::new(buf);
    if r.bits(32)? != SYNC_EXSS {
        return Err(Error::NoSync);
    }
    let _user = r.bits(8)?;
    let index = r.bits(2)?;
    let long = r.flag()?;
    let (bh, bf) = if long { (12, 20) } else { (8, 16) };
    let header_size = r.bits(bh)? as usize + 1;
    let size = r.bits(bf)? as usize + 1;
    if header_size > size || size > buf.len() {
        return Err(Error::Truncated { at_bit: buf.len() * 8, wanted: ((size.saturating_sub(buf.len())) * 8) as u32 });
    }
    if header_size < 7 || crc16(&buf[5..header_size]) != 0 {
        return Err(Error::Invalid("extension substream header CRC"));
    }
    let static_fields = r.flag()?;
    let mut nassets = 1usize;
    let mut mix_enabled = false;
    let mut mix_out_ch: Vec<u32> = Vec::new();
    let mut duration = None;
    if static_fields {
        let _ref_clock = r.bits(2)?;
        duration = Some(512 * (r.bits(3)? + 1));
        if r.flag()? {
            r.bits(32)?;
            r.bits(4)?;
        }
        let npres = r.bits(3)? as usize + 1;
        nassets = r.bits(3)? as usize + 1;
        let mut active_ss = vec![0u32; npres];
        for m in active_ss.iter_mut() {
            *m = r.bits(index + 1)?;
        }
        for m in &active_ss {
            for ss in 0..=index {
                if (m >> ss) & 1 == 1 {
                    let _asset_mask = r.bits(8)?;
                }
            }
        }
        mix_enabled = r.flag()?;
        if mix_enabled {
            let _adj_level = r.bits(2)?;
            let bits4mix = (r.bits(2)? + 1) << 2;
            let nconfigs = r.bits(2)? as usize + 1;
            for _ in 0..nconfigs {
                mix_out_ch.push(num_speakers(r.bits(bits4mix)?));
            }
        }
    }
    let mut sizes = Vec::with_capacity(nassets);
    for _ in 0..nassets {
        sizes.push(r.bits(bf)? as usize + 1);
    }
    let mut assets = Vec::with_capacity(nassets);
    let mut data_off = header_size;
    for &asize in &sizes {
        let start = r.position_bits();
        let desc_size = r.bits(9)? as usize + 1;
        let mut a = parse_asset(&mut r, static_fields, mix_enabled, &mix_out_ch, bf, duration)?;
        r.seek_bits(start + desc_size * 8)?;
        // Components follow one another in mask-bit order (Table 7-15),
        // inside this asset's nuAssetFsize bytes.
        let mut off = data_off;
        let mut comps = Vec::new();
        for (bit, len) in std::mem::take(&mut a.components).into_iter().map(|(b, r)| (b, r.end)) {
            comps.push((bit, off..off + len));
            off += len;
        }
        if off > data_off + asize || data_off + asize > size {
            return Err(Error::Invalid("extension substream components overrun their asset"));
        }
        a.components = comps;
        assets.push(a);
        data_off += asize;
    }
    Ok(ExssFrame { index, size, duration, assets, bits4fsize: bf })
}

/// One audio asset descriptor; `components` come back as `(bit, 0..len)`.
fn parse_asset(
    r: &mut BitReader,
    static_fields: bool,
    mix_enabled: bool,
    mix_out_ch: &[u32],
    bf: u32,
    duration: Option<u32>,
) -> Result<Asset, Error> {
    let mut a = Asset { index: r.bits(3)?, ..Default::default() };
    let mut one2one = false;
    let (mut emb_stereo, mut emb_six) = (false, false);
    if static_fields {
        if r.flag()? {
            r.bits(4)?;
        }
        if r.flag()? {
            r.bits(24)?;
        }
        if r.flag()? {
            let n = r.bits(10)? + 1;
            for _ in 0..n {
                r.bits(8)?;
            }
        }
        a.bit_resolution = Some(r.bits(5)? + 1);
        a.max_sample_rate = Some(SAMPLE_RATES_7_9[r.bits(4)? as usize]);
        a.total_channels = r.bits(8)? + 1;
        one2one = r.flag()?;
        if one2one {
            if a.total_channels > 2 {
                emb_stereo = r.flag()?;
            }
            if a.total_channels > 6 {
                emb_six = r.flag()?;
            }
            let mut bits4mask = 0;
            if r.flag()? {
                bits4mask = (r.bits(2)? + 1) << 2;
                a.spkr_mask = Some(r.bits(bits4mask)?);
            }
            let nremap = r.bits(3)? as usize;
            let mut layouts = Vec::with_capacity(nremap);
            for _ in 0..nremap {
                layouts.push(r.bits(bits4mask)?);
            }
            for l in layouts {
                let nspk = num_speakers(l);
                let ndec = r.bits(5)? + 1;
                for _ in 0..nspk {
                    let m = r.bits(ndec)?;
                    for _ in 0..m.count_ones() {
                        r.bits(5)?;
                    }
                }
            }
        } else {
            let _representation = r.bits(3)?;
        }
    }
    // Dynamic metadata (Table 7-6).
    let drc = r.flag()?;
    if drc {
        r.bits(8)?;
    }
    if r.flag()? {
        r.bits(5)?;
    }
    if drc && emb_stereo {
        r.bits(8)?;
    }
    let mix_present = mix_enabled && r.flag()?;
    if mix_present {
        let _external = r.flag()?;
        let _post_gain = r.bits(6)?;
        let control = r.bits(2)?;
        if control < 3 {
            r.bits(3)?;
        } else {
            r.bits(8)?;
        }
        let per_ch = r.flag()?;
        for &n in mix_out_ch {
            if per_ch {
                for _ in 0..n {
                    r.bits(6)?;
                }
            } else {
                r.bits(6)?;
            }
        }
        let mut dec_ch = vec![a.total_channels];
        if emb_six {
            dec_ch.push(6);
        }
        if emb_stereo {
            dec_ch.push(2);
        }
        for &n in mix_out_ch {
            for &nd in &dec_ch {
                for _ in 0..nd {
                    let m = r.bits(n)?;
                    for _ in 0..m.count_ones() {
                        r.bits(6)?;
                    }
                }
            }
        }
    }
    // Decoder navigation data (Table 7-7).
    a.coding_mode = r.bits(2)?;
    let mut comps: Vec<(u32, Range<usize>)> = Vec::new();
    let xll = |r: &mut BitReader, comps: &mut Vec<(u32, Range<usize>)>| -> Result<(), Error> {
        let len = r.bits(bf)? as usize + 1;
        comps.push((mask::EXSS_XLL, 0..len));
        if r.flag()? {
            r.bits(4)?;
            let nb = r.bits(5)? + 1;
            r.bits(nb)?;
            r.bits(bf)?;
        }
        Ok(())
    };
    match a.coding_mode {
        0 => {
            a.ext_mask = r.bits(12)?;
            if a.ext_mask & mask::EXSS_CORE != 0 {
                comps.push((mask::EXSS_CORE, 0..r.bits(14)? as usize + 1));
                if r.flag()? {
                    r.bits(2)?;
                }
            }
            if a.ext_mask & mask::EXSS_XBR != 0 {
                comps.push((mask::EXSS_XBR, 0..r.bits(14)? as usize + 1));
            }
            if a.ext_mask & mask::EXSS_XXCH != 0 {
                comps.push((mask::EXSS_XXCH, 0..r.bits(14)? as usize + 1));
            }
            if a.ext_mask & mask::EXSS_X96 != 0 {
                comps.push((mask::EXSS_X96, 0..r.bits(12)? as usize + 1));
            }
            if a.ext_mask & mask::EXSS_LBR != 0 {
                comps.push((mask::EXSS_LBR, 0..r.bits(14)? as usize + 1));
                if r.flag()? {
                    r.bits(2)?;
                }
            }
            if a.ext_mask & mask::EXSS_XLL != 0 {
                xll(r, &mut comps)?;
            }
            if a.ext_mask & 0x400 != 0 {
                r.bits(16)?;
            }
            if a.ext_mask & 0x800 != 0 {
                r.bits(16)?;
            }
        }
        1 => {
            a.ext_mask = mask::EXSS_XLL;
            xll(r, &mut comps)?;
        }
        2 => {
            a.ext_mask = mask::EXSS_LBR;
            comps.push((mask::EXSS_LBR, 0..r.bits(14)? as usize + 1));
            if r.flag()? {
                r.bits(2)?;
            }
        }
        _ => {
            let len = r.bits(14)? as usize + 1;
            let _codec = r.bits(8)?;
            if r.flag()? {
                r.bits(3)?;
            }
            comps.push((0, 0..len));
        }
    }
    if a.ext_mask & mask::EXSS_XLL != 0 {
        a.stream_id = Some(r.bits(3)?);
    }
    if one2one && mix_enabled && !mix_present && r.flag()? {
        let per_ch = r.flag()?;
        for &n in mix_out_ch {
            if per_ch {
                for _ in 0..n {
                    r.bits(6)?;
                }
            } else {
                r.bits(6)?;
            }
        }
    }
    let _secondary = r.flag()?;
    if r.flag()? {
        r.bits(4)?;
        if let Some(d) = duration {
            for _ in 0..d / 256 {
                r.bits(8)?;
            }
        }
    }
    a.components = comps;
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_counts_follow_table_7_10() {
        assert_eq!(num_speakers(0x000F), 6, "C LR LsRs LFE1");
        assert_eq!(num_speakers(0x004F), 8, "+ LsrRsr");
        assert_eq!(num_speakers(0x0010), 1, "Cs");
    }
}
