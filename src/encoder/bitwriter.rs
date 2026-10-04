//! MSB-first bit writer: the mirror of the decoder's `BitReader`.

/// Accumulates bits most significant first into bytes.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits used in the last byte (0 when byte aligned).
    used: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bits written so far.
    pub fn len_bits(&self) -> usize {
        if self.used == 0 {
            self.bytes.len() * 8
        } else {
            (self.bytes.len() - 1) * 8 + self.used as usize
        }
    }

    /// Write the low `n` bits (0..=32) of `v`, most significant first.
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(n == 32 || v >> n == 0, "{v:#x} does not fit in {n} bits");
        for i in (0..n).rev() {
            if self.used == 0 {
                self.bytes.push(0);
            }
            let bit = ((v >> i) & 1) as u8;
            *self.bytes.last_mut().expect("pushed above") |= bit << (7 - self.used);
            self.used = (self.used + 1) & 7;
        }
    }

    /// Write `v` as an `n`-bit two's-complement field.
    pub fn put_signed(&mut self, v: i32, n: u32) {
        debug_assert!((1..=32).contains(&n));
        debug_assert!(
            n == 32 || ((v as i64) >= -(1i64 << (n - 1)) && (v as i64) < (1i64 << (n - 1))),
            "{v} does not fit in {n} signed bits"
        );
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.put(v as u32 & mask, n);
    }

    pub fn flag(&mut self, b: bool) {
        self.put(u32::from(b), 1);
    }

    /// The bytes, zero-padded to a byte boundary.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unusual_byte_groupings)] // grouped as the writes split them
    fn writes_msb_first_across_bytes() {
        let mut w = BitWriter::new();
        w.put(0b101, 3);
        w.put(0b0_1100_01, 7);
        w.put(0b01_0011, 6);
        assert_eq!(w.len_bits(), 16);
        w.put(0xFF, 8);
        w.put_signed(-2, 2);
        assert_eq!(w.len_bits(), 26);
        assert_eq!(
            w.into_bytes(),
            vec![0b1010_1100, 0b0101_0011, 0xFF, 0b1000_0000]
        );
    }

    #[test]
    fn signed_fields_read_back_through_the_decoder_convention() {
        let mut w = BitWriter::new();
        for v in [-16, 15, 0, -1, 7] {
            w.put_signed(v, 5);
        }
        w.put(0xDEAD_BEEF, 32);
        let b = w.into_bytes();
        let mut r = crate::bits::BitReader::new(&b);
        for want in [-16, 15, 0, -1, 7] {
            assert_eq!(r.sbits(5).unwrap(), want);
        }
        assert_eq!(r.bits(32).unwrap(), 0xDEAD_BEEF);
    }
}
