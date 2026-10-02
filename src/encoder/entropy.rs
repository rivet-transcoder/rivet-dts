//! Encode side of the core's entropy codes: the Annex D.5 Huffman books
//! indexed by level, the D.6 block codes and plain two's complement, and
//! the (`ABITS`, `SEL`) choices of Table 5-26.

use std::sync::LazyLock;

use super::bitwriter::BitWriter;
use crate::tables::{self, HuffEntry};

/// A Huffman book for encoding: `(length, code)` by level.
pub struct EncBook {
    min: i32,
    codes: Vec<(u8, u32)>,
}

impl EncBook {
    fn new(entries: &[HuffEntry]) -> Self {
        let min = entries.iter().map(|e| e.0).min().expect("non-empty book");
        let max = entries.iter().map(|e| e.0).max().expect("non-empty book");
        let mut codes = vec![(0u8, 0u32); (max - min + 1) as usize];
        for &(level, len, code) in entries {
            codes[(level - min) as usize] = (len, code);
        }
        Self { min, codes }
    }

    /// Code length of `level`, or `None` outside the book.
    pub fn len(&self, level: i32) -> Option<u32> {
        let i = level - self.min;
        if i < 0 {
            return None;
        }
        self.codes.get(i as usize).filter(|c| c.0 > 0).map(|c| c.0 as u32)
    }

    pub fn put(&self, w: &mut BitWriter, level: i32) {
        let (len, code) = self.codes[(level - self.min) as usize];
        debug_assert!(len > 0, "level {level} not in book");
        w.put(code, len as u32);
    }
}

macro_rules! books {
    ($($name:ident = $table:ident),* $(,)?) => {
        $(static $name: LazyLock<EncBook> = LazyLock::new(|| EncBook::new(&tables::$table));)*
    };
}

books!(
    A3 = HUFF_A3, A4 = HUFF_A4, A5 = HUFF_A5, B5 = HUFF_B5, C5 = HUFF_C5, A7 = HUFF_A7, B7 = HUFF_B7,
    C7 = HUFF_C7, A9 = HUFF_A9, B9 = HUFF_B9, C9 = HUFF_C9, A13 = HUFF_A13, B13 = HUFF_B13, C13 = HUFF_C13,
    A17 = HUFF_A17, B17 = HUFF_B17, C17 = HUFF_C17, D17 = HUFF_D17, E17 = HUFF_E17, F17 = HUFF_F17,
    G17 = HUFF_G17, A25 = HUFF_A25, B25 = HUFF_B25, C25 = HUFF_C25, D25 = HUFF_D25, E25 = HUFF_E25,
    F25 = HUFF_F25, G25 = HUFF_G25, A33 = HUFF_A33, B33 = HUFF_B33, C33 = HUFF_C33, D33 = HUFF_D33,
    E33 = HUFF_E33, F33 = HUFF_F33, G33 = HUFF_G33, A65 = HUFF_A65, B65 = HUFF_B65, C65 = HUFF_C65,
    D65 = HUFF_D65, E65 = HUFF_E65, F65 = HUFF_F65, G65 = HUFF_G65, A129 = HUFF_A129, B129 = HUFF_B129,
    C129 = HUFF_C129, D129 = HUFF_D129, E129 = HUFF_E129, F129 = HUFF_F129, G129 = HUFF_G129,
    SA129 = HUFF_SA129, SB129 = HUFF_SB129, SC129 = HUFF_SC129, SD129 = HUFF_SD129, SE129 = HUFF_SE129,
);

/// The `TMODE` book for `THUFF` = 0 (A4), the only one the encoder uses.
pub fn tmode_book() -> &'static EncBook {
    &A4
}

/// Table 5-24: the scale-factor difference books for `SHUFF` 0..=4.
pub fn scale_book(shuff: u32) -> &'static EncBook {
    match shuff {
        0 => &SA129,
        1 => &SB129,
        2 => &SC129,
        3 => &SD129,
        _ => &SE129,
    }
}

/// How one subband subsubframe's eight indices are written.
#[derive(Clone, Copy)]
pub enum Coding {
    Huffman(&'static EncBook),
    /// Two block codes of `bits` bits, each four indices of `levels` levels.
    Block { levels: u32, bits: u32 },
    /// `bits`-bit two's complement per index.
    Raw { bits: u32 },
}

impl Coding {
    pub fn is_huffman(self) -> bool {
        matches!(self, Coding::Huffman(_))
    }

    /// Bits for eight indices, or `None` if one is out of range.
    pub fn cost(self, q: &[i32]) -> Option<u32> {
        match self {
            Coding::Huffman(b) => q.iter().map(|v| b.len(*v)).sum(),
            Coding::Block { levels, bits } => {
                let h = ((levels - 1) / 2) as i32;
                q.iter().all(|v| v.abs() <= h).then_some(bits * (q.len() as u32 / 4))
            }
            Coding::Raw { bits } => {
                let lim = 1i32 << (bits - 1);
                q.iter().all(|v| *v >= -lim && *v < lim).then_some(bits * q.len() as u32)
            }
        }
    }

    pub fn put(self, w: &mut BitWriter, q: &[i32]) {
        match self {
            Coding::Huffman(b) => q.iter().for_each(|v| b.put(w, *v)),
            Coding::Block { levels, bits } => {
                let h = ((levels - 1) / 2) as i32;
                for chunk in q.chunks(4) {
                    // Annex C.3.2: code = Σ (index + offset)·L^i, first element least significant.
                    let code = chunk.iter().rev().fold(0u32, |acc, v| acc * levels + (v + h) as u32);
                    w.put(code, bits);
                }
            }
            Coding::Raw { bits } => q.iter().for_each(|v| w.put_signed(*v, bits)),
        }
    }
}

/// Table 5-26: the codings selectable for `ABITS` = `abits` (1..=26), by
/// `SEL`. For `ABITS` > 10 there is no `SEL` and one coding.
pub fn codings(abits: u32) -> &'static [Coding] {
    static ALL: LazyLock<Vec<Vec<Coding>>> = LazyLock::new(|| {
        let block = |levels: u32| {
            let bits = tables::BLOCK_CODE_BITS
                .iter()
                .find(|(l, _)| *l == levels)
                .map(|(_, b)| *b as u32)
                .expect("block code width");
            Coding::Block { levels, bits }
        };
        let h = |b: &'static LazyLock<EncBook>| Coding::Huffman(LazyLock::force(b));
        let mut v: Vec<Vec<Coding>> = vec![Vec::new(); 27];
        v[1] = vec![h(&A3), block(3)];
        v[2] = vec![h(&A5), h(&B5), h(&C5), block(5)];
        v[3] = vec![h(&A7), h(&B7), h(&C7), block(7)];
        v[4] = vec![h(&A9), h(&B9), h(&C9), block(9)];
        v[5] = vec![h(&A13), h(&B13), h(&C13), block(13)];
        v[6] = vec![h(&A17), h(&B17), h(&C17), h(&D17), h(&E17), h(&F17), h(&G17), block(17)];
        v[7] = vec![h(&A25), h(&B25), h(&C25), h(&D25), h(&E25), h(&F25), h(&G25), block(25)];
        v[8] = vec![h(&A33), h(&B33), h(&C33), h(&D33), h(&E33), h(&F33), h(&G33), Coding::Raw { bits: 5 }];
        v[9] = vec![h(&A65), h(&B65), h(&C65), h(&D65), h(&E65), h(&F65), h(&G65), Coding::Raw { bits: 6 }];
        v[10] = vec![
            h(&A129),
            h(&B129),
            h(&C129),
            h(&D129),
            h(&E129),
            h(&F129),
            h(&G129),
            Coding::Raw { bits: 7 },
        ];
        for (a, item) in v.iter_mut().enumerate().skip(11) {
            *item = vec![Coding::Raw { bits: a as u32 - 3 }];
        }
        v
    });
    &ALL[abits as usize]
}

/// Largest index magnitude the encoder quantises to for `abits`, chosen so
/// every coding of Table 5-26 for that `ABITS` can carry it (the 2ⁿ-level
/// linear forms of `ABITS` 8–10 stop one short of the Huffman range).
pub fn max_index(abits: u32) -> i32 {
    match abits {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 6,
        6 => 8,
        7 => 12,
        8 => 15,
        9 => 31,
        10 => 63,
        a => (1 << (a - 4)) - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::huffman::{self, SampleCoding};

    /// Every coding of every `ABITS` writes indices that the decoder's own
    /// Table 5-26 resolution reads back.
    #[test]
    fn every_coding_round_trips_through_the_decoder() {
        for abits in 1..=26u32 {
            let m = max_index(abits);
            let q: Vec<i32> = (0..8).map(|i| [m, -m, 0, 1, -1, m / 2, -(m / 3), m - 1][i]).collect();
            for (sel, c) in codings(abits).iter().enumerate() {
                let mut w = BitWriter::new();
                c.put(&mut w, &q);
                let cost = c.cost(&q).expect("in range");
                assert_eq!(w.len_bits(), cost as usize, "ABITS {abits} SEL {sel}");
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                let sel_field = if abits <= 10 { sel as u32 } else { 0 };
                let mut got = [0i32; 8];
                match huffman::sample_coding(abits, sel_field).unwrap() {
                    SampleCoding::Huffman(b) => got.iter_mut().for_each(|v| *v = b.decode(&mut r).unwrap()),
                    SampleCoding::Raw { bits } => got.iter_mut().for_each(|v| *v = r.sbits(bits).unwrap()),
                    SampleCoding::Block { levels, bits } => {
                        let mut blk = [0i32; 4];
                        for half in 0..2 {
                            huffman::decode_block(r.bits(bits).unwrap(), levels, &mut blk).unwrap();
                            got[half * 4..half * 4 + 4].copy_from_slice(&blk);
                        }
                    }
                    SampleCoding::None => unreachable!(),
                }
                assert_eq!(&got[..], &q[..], "ABITS {abits} SEL {sel}");
            }
        }
    }

    #[test]
    fn scale_and_tmode_books_round_trip() {
        for shuff in 0..5 {
            let mut w = BitWriter::new();
            let levels: Vec<i32> = (-64..=64).collect();
            levels.iter().for_each(|l| scale_book(shuff).put(&mut w, *l));
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let dec = huffman::scale_book(shuff).unwrap();
            for l in &levels {
                assert_eq!(dec.decode(&mut r).unwrap(), *l);
            }
        }
        let mut w = BitWriter::new();
        (0..4).for_each(|l| tmode_book().put(&mut w, l));
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for l in 0..4 {
            assert_eq!(huffman::tmode_book(0).decode(&mut r).unwrap(), l);
        }
    }
}
