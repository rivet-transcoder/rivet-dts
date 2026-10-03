//! Speakers and the channel layouts the decoder outputs.
//!
//! The core names its channels by `AMODE` (Table 5-4); the XXCH extension
//! and the extension substream by loudspeaker masks (Tables 6-22, 7-10).
//! Both are mapped onto one [`Speaker`] vocabulary (the conventional short
//! names: FL, FR, FC, LFE, …) and the output is always in the canonical
//! order of that vocabulary — the order of the [`Speaker`] variants, which
//! is the WAVE (`WAVEFORMATEXTENSIBLE` channel mask) order — so a decoded frame is a [`Layout`] with nothing left implicit.

use std::fmt;

/// A speaker position, by its conventional short name; the declaration
/// order is the order channels are interleaved in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Speaker {
    /// Front left (core L).
    FL,
    /// Front right (core R).
    FR,
    /// Front centre (core C; also mono `AMODE` 0).
    FC,
    /// Low-frequency effects (LFE1).
    LFE,
    /// Back left (XXCH `Lsr`, left surround in rear, −150°).
    BL,
    /// Back right (XXCH `Rsr`).
    BR,
    /// Front left of centre (XXCH `Lc`).
    FLC,
    /// Front right of centre (XXCH `Rc`).
    FRC,
    /// Back centre (the single surround of `AMODE` 6/7; XCh/XXCH `Cs`).
    BC,
    /// Side left (core `Ls`; XXCH `Lss`).
    SL,
    /// Side right (core `Rs`; XXCH `Rss`).
    SR,
    /// Top centre (XXCH `Oh`, over the listener's head).
    TC,
    /// Top front left (XXCH `Lh`).
    TFL,
    /// Top front centre (XXCH `Ch`).
    TFC,
    /// Top front right (XXCH `Rh`).
    TFR,
    /// Top back left (XXCH `Lhr`).
    TBL,
    /// Top back centre (XXCH `Chr`).
    TBC,
    /// Top back right (XXCH `Rhr`).
    TBR,
    /// Wide left (XXCH `Lw`, −60°).
    WL,
    /// Wide right (XXCH `Rw`).
    WR,
    /// Second low-frequency effects (XXCH `LFE2`).
    LFE2,
    /// Top side left (XXCH `Lhs`).
    TSL,
    /// Top side right (XXCH `Rhs`).
    TSR,
    /// Bottom front centre (XXCH `Cl`).
    BFC,
    /// Bottom front left (XXCH `Ll`).
    BFL,
    /// Bottom front right (XXCH `Rl`).
    BFR,
}

/// Every [`Speaker`], in order.
const ALL: [Speaker; 26] = {
    use Speaker::*;
    [
        FL, FR, FC, LFE, BL, BR, FLC, FRC, BC, SL, SR, TC, TFL, TFC, TFR, TBL, TBC, TBR, WL, WR, LFE2, TSL, TSR, BFC,
        BFL, BFR,
    ]
};

/// A set of speakers, kept in canonical order: the channels of a
/// [`Layout::Custom`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerList {
    len: u8,
    list: [Speaker; 26],
}

impl SpeakerList {
    /// The set of `speakers` (duplicates collapse), in canonical order.
    pub fn new(speakers: &[Speaker]) -> Self {
        let mut list = [Speaker::FL; 26];
        let mut len = 0;
        for s in ALL {
            if speakers.contains(&s) {
                list[len] = s;
                len += 1;
            }
        }
        Self { len: len as u8, list }
    }

    /// The speakers, in canonical order.
    pub fn as_slice(&self) -> &[Speaker] {
        &self.list[..self.len as usize]
    }
}

impl fmt::Debug for SpeakerList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// The channel layouts the decoder outputs: the common ones by their
/// conventional name (`stereo`, `5.1(side)`, …), anything else as a [`Custom`](Layout::Custom) list. The channels
/// of every layout are in canonical [`Speaker`] order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Layout {
    /// `mono`: FC (`AMODE` 0).
    Mono,
    /// `stereo`: FL FR (`AMODE` 1–4: dual mono, L+R, sum/difference, LT+RT).
    Stereo,
    /// `2.1`: FL FR LFE (`AMODE` 1–4 with the LFE).
    Stereo21,
    /// `3.0`: FL FR FC (`AMODE` 5, C + L + R).
    Surround30,
    /// `3.0(back)`: FL FR BC (`AMODE` 6, L + R + S).
    Surround30Back,
    /// `3.1`: FL FR FC LFE (`AMODE` 5 with the LFE).
    Surround31,
    /// `4.0`: FL FR FC BC (`AMODE` 7, C + L + R + S).
    Surround40,
    /// `4.1`: FL FR FC LFE BC (`AMODE` 7 with the LFE).
    Surround41,
    /// `quad(side)`: FL FR SL SR (`AMODE` 8).
    QuadSide,
    /// `5.0(side)`: FL FR FC SL SR (`AMODE` 9).
    Surround50Side,
    /// `5.1(side)`: FL FR FC LFE SL SR (`AMODE` 9 with the LFE).
    Surround51Side,
    /// `6.0`: FL FR FC BC SL SR (5.0 core + XCh back centre).
    Surround60,
    /// `6.1`: FL FR FC LFE BC SL SR (5.1 core + XCh back centre).
    Surround61,
    /// `7.0`: FL FR FC BL BR SL SR.
    Surround70,
    /// `7.1`: FL FR FC LFE BL BR SL SR (5.1 core + XXCH rear pair).
    Surround71,
    /// Any other set of speakers, in canonical order.
    Custom(SpeakerList),
}

const NAMED: [Layout; 15] = [
    Layout::Mono,
    Layout::Stereo,
    Layout::Stereo21,
    Layout::Surround30,
    Layout::Surround30Back,
    Layout::Surround31,
    Layout::Surround40,
    Layout::Surround41,
    Layout::QuadSide,
    Layout::Surround50Side,
    Layout::Surround51Side,
    Layout::Surround60,
    Layout::Surround61,
    Layout::Surround70,
    Layout::Surround71,
];

impl Layout {
    /// The layout of a set of speakers: a named one when it is one, else
    /// [`Custom`](Layout::Custom).
    pub fn from_speakers(speakers: &[Speaker]) -> Layout {
        let list = SpeakerList::new(speakers);
        NAMED
            .iter()
            .copied()
            .find(|l| l.speakers() == list.as_slice())
            .unwrap_or(Layout::Custom(list))
    }

    /// The layout's conventional name (`"5.1(side)"`, `"stereo"`, …), or
    /// `"custom"`.
    pub fn name(&self) -> &'static str {
        match self {
            Layout::Mono => "mono",
            Layout::Stereo => "stereo",
            Layout::Stereo21 => "2.1",
            Layout::Surround30 => "3.0",
            Layout::Surround30Back => "3.0(back)",
            Layout::Surround31 => "3.1",
            Layout::Surround40 => "4.0",
            Layout::Surround41 => "4.1",
            Layout::QuadSide => "quad(side)",
            Layout::Surround50Side => "5.0(side)",
            Layout::Surround51Side => "5.1(side)",
            Layout::Surround60 => "6.0",
            Layout::Surround61 => "6.1",
            Layout::Surround70 => "7.0",
            Layout::Surround71 => "7.1",
            Layout::Custom(_) => "custom",
        }
    }

    /// The speakers, in the order the interleaved samples carry them.
    pub fn speakers(&self) -> &[Speaker] {
        use Speaker::*;
        match self {
            Layout::Mono => &[FC],
            Layout::Stereo => &[FL, FR],
            Layout::Stereo21 => &[FL, FR, LFE],
            Layout::Surround30 => &[FL, FR, FC],
            Layout::Surround30Back => &[FL, FR, BC],
            Layout::Surround31 => &[FL, FR, FC, LFE],
            Layout::Surround40 => &[FL, FR, FC, BC],
            Layout::Surround41 => &[FL, FR, FC, LFE, BC],
            Layout::QuadSide => &[FL, FR, SL, SR],
            Layout::Surround50Side => &[FL, FR, FC, SL, SR],
            Layout::Surround51Side => &[FL, FR, FC, LFE, SL, SR],
            Layout::Surround60 => &[FL, FR, FC, BC, SL, SR],
            Layout::Surround61 => &[FL, FR, FC, LFE, BC, SL, SR],
            Layout::Surround70 => &[FL, FR, FC, BL, BR, SL, SR],
            Layout::Surround71 => &[FL, FR, FC, LFE, BL, BR, SL, SR],
            Layout::Custom(l) => l.as_slice(),
        }
    }

    /// Channel count.
    pub fn channels(&self) -> usize {
        self.speakers().len()
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::Custom(l) => write!(f, "custom{:?}", l),
            _ => f.write_str(self.name()),
        }
    }
}

/// The core's channels for `AMODE` 0–9, in coded order (Table 5-4).
pub(crate) fn amode_speakers(amode: u32) -> Option<&'static [Speaker]> {
    use Speaker::*;
    Some(match amode {
        0 => &[FC],
        1..=4 => &[FL, FR],
        5 => &[FC, FL, FR],
        6 => &[FL, FR, BC],
        7 => &[FC, FL, FR, BC],
        8 => &[FL, FR, SL, SR],
        9 => &[FC, FL, FR, SL, SR],
        _ => return None,
    })
}

/// The speaker of each bit of the XXCH / core activity masks (Table 6-22),
/// `None` for LFE1 (carried by the core LFE) and reserved bits.
pub(crate) fn xxch_mask_speaker(bit: u32) -> Option<Speaker> {
    use Speaker::*;
    Some(match bit {
        0 => FC,
        1 => FL,
        2 => FR,
        3 => SL,
        4 => SR,
        5 => LFE,
        6 => BC,
        7 => BL,
        8 => BR,
        9 => SL,
        10 => SR,
        11 => FLC,
        12 => FRC,
        13 => TFL,
        14 => TFC,
        15 => TFR,
        16 => LFE2,
        17 => WL,
        18 => WR,
        19 => TC,
        20 => TSL,
        21 => TSR,
        22 => TBC,
        23 => TBL,
        24 => TBR,
        25 => BFC,
        26 => BFL,
        27 => BFR,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_layouts_are_in_canonical_order_and_found_again() {
        for l in NAMED {
            let s = l.speakers();
            assert!(s.windows(2).all(|w| w[0] < w[1]), "{l}: not canonical");
            assert_eq!(Layout::from_speakers(s), l);
            // Any order in, the same layout out.
            let mut rev = s.to_vec();
            rev.reverse();
            assert_eq!(Layout::from_speakers(&rev), l);
        }
        let odd = Layout::from_speakers(&[Speaker::FC, Speaker::LFE]);
        assert!(matches!(odd, Layout::Custom(_)));
        assert_eq!(odd.speakers(), &[Speaker::FC, Speaker::LFE]);
        assert_eq!(odd.name(), "custom");
    }
}
