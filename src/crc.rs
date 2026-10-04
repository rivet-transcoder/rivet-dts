//! The CRC of Annex B: CRC-16-CCITT, G(x) = x^16 + x^12 + x^5 + 1, MSB
//! first, initialised to 0xFFFF. Every DTS-HD header carries one at its
//! end, so running it over the covered bytes *and* the stored CRC leaves 0.

/// CRC-16 of `data` per Annex B.
pub(crate) fn crc16(data: &[u8]) -> u16 {
    let mut c: u16 = 0xFFFF;
    for &b in data {
        c ^= (b as u16) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 {
                (c << 1) ^ 0x1021
            } else {
                c << 1
            };
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccitt_check_value_and_residue() {
        // The CRC-16/CCITT-FALSE check value of "123456789".
        assert_eq!(crc16(b"123456789"), 0x29B1);
        // Appending the CRC (big-endian) leaves a zero remainder, which is
        // how the headers are verified.
        let mut d = b"DTS-HD header".to_vec();
        let c = crc16(&d);
        d.extend_from_slice(&c.to_be_bytes());
        assert_eq!(crc16(&d), 0);
    }
}
