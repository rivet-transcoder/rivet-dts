//! Decoder-level tests of the frame-level contracts, on synthetic frames.
//! Round trips through the encoder are in `tests/encoder_roundtrip.rs`,
//! public sample streams in `tests/samples.rs`.

use super::*;

/// A synthetic bit-stream header (Table 5-1 fields only), for the refusals
/// that trigger before any audio is parsed.
fn header(ftype: u32, nblks: u32, fsize: u32, amode: u32, sfreq: u32, lff: u32, vernum: u32) -> Vec<u8> {
    pack(&header_bits(ftype, nblks, fsize, amode, sfreq, lff, vernum), fsize)
}

fn pack(bits: &[u8], fsize: u32) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        out[i / 8] |= b << (7 - (i % 8));
    }
    out.resize(fsize as usize + 1, 0);
    out
}

/// The Table 5-1 header followed by `SUBFS` = 0 and `PCHS`, as bits.
fn header_bits(ftype: u32, nblks: u32, fsize: u32, amode: u32, sfreq: u32, lff: u32, vernum: u32) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |v: u32, n: usize| {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    };
    push(CORE_SYNC, 32);
    push(ftype, 1);
    push(31, 5); // SHORT
    push(0, 1); // CPF
    push(nblks, 7);
    push(fsize, 14);
    push(amode, 6);
    push(sfreq, 4);
    push(24, 5); // RATE
    push(0, 1); // FixedBit
    push(0, 4); // DYNF TIMEF AUXF HDCD
    push(0, 3); // EXT_AUDIO_ID
    push(0, 1); // EXT_AUDIO
    push(0, 1); // ASPF
    push(lff, 2);
    push(1, 1); // HFLAG
    push(0, 1); // FILTS
    push(vernum, 4);
    push(0, 2); // CHIST
    push(0, 3); // PCMR
    push(0, 2); // SUMF SUMS
    push(0, 4); // DIALNORM
    // Primary audio coding header: SUBFS=0, PCHS=channels-1, then zeros.
    push(0, 4);
    push(AMODE_CHANNELS[amode as usize] as u32 - 1, 3);
    bits
}

/// A 5.1 frame whose first subframe predicts its first subband (`PMODE` = 1).
/// Every other header field is 0, which is valid: two active subbands per
/// channel, none VQ, no joint coding, code books A, `SEL` 0 everywhere (so
/// every `ADJ` is transmitted), one subsubframe.
fn frame_with_adpcm() -> Vec<u8> {
    let mut bits = header_bits(1, 15, 2047, 9, 13, 2, 7);
    // SUBS 5×5, VQSUB 5×5, JOINX 5×3, THUFF 5×2, SHUFF 5×3, BHUFF 5×3, SEL
    // 5×(1 + 4×2 + 5×3), ADJ 5×(2 + 4×2 + 5×2), then SSC 2 + PSC 3.
    bits.extend(std::iter::repeat_n(0, 25 + 25 + 15 + 10 + 15 + 15 + 120 + 100 + 5));
    bits.push(1); // PMODE[0][0]
    bits.extend(std::iter::repeat_n(0, 9 + 12)); // the other PMODEs, PVQ[0][0]
    pack(&bits, 2047)
}

#[test]
fn adpcm_prediction_is_refused_by_name() {
    let mut d = Decoder::new();
    let err = d.decode(&frame_with_adpcm()).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("ADPCM prediction (PMODE = 1 in 1 subbands)"), "{err}");
    assert!(err.to_string().contains("D.10.1"), "{err}");
}

#[test]
fn rejects_non_dts_packets_by_name() {
    let mut d = Decoder::new();
    let err = d.decode(&[0x0B, 0x77, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap_err();
    assert!(err.to_string().contains("0x7FFE8001"), "{err}");
}

#[test]
fn a_short_packet_is_truncated_not_zero_filled() {
    let mut d = Decoder::new();
    let f = header(1, 15, 2047, 9, 13, 2, 7);
    let err = d.decode(&f[..100]).unwrap_err();
    assert!(err.to_string().contains("truncated"), "{err}");
}

#[test]
fn termination_frames_are_refused_by_name() {
    let mut d = Decoder::new();
    let f = header(0, 15, 2047, 9, 13, 2, 7);
    let err = d.decode(&f).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("termination frame"), "{err}");
}

#[test]
fn incompatible_encoder_revisions_are_refused_by_name() {
    let mut d = Decoder::new();
    let f = header(1, 15, 2047, 9, 13, 2, 9);
    let err = d.decode(&f).unwrap_err();
    assert!(err.to_string().contains("VERNUM 9"), "{err}");
}

#[test]
fn arrangements_without_defined_speakers_are_refused_by_name() {
    // AMODE 10 = CL + CR + L + R + SL + SR: six channels, of which the core
    // carries five without saying which.
    let mut d = Decoder::new();
    let f = header(1, 15, 2047, 10, 13, 0, 7);
    let err = d.decode(&f).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("AMODE 10"), "{err}");
    assert!(core_speakers(33).unwrap_err().to_string().contains("user-defined"));
}

#[test]
fn every_core_arrangement_maps_onto_canonical_order() {
    use Speaker::*;
    let layout = |amode: u32, lfe: bool| {
        let mut s = layout::amode_speakers(amode).unwrap().to_vec();
        if lfe {
            s.push(LFE);
        }
        Layout::from_speakers(&s)
    };
    assert_eq!(layout(9, true), Layout::Surround51Side);
    assert_eq!(layout(9, true).speakers(), &[FL, FR, FC, LFE, SL, SR]);
    assert_eq!(layout(2, true), Layout::Stereo21);
    assert_eq!(layout(0, false), Layout::Mono);
    assert_eq!(layout(5, true), Layout::Surround31);
    assert_eq!(layout(6, false), Layout::Surround30Back);
    assert_eq!(layout(7, true), Layout::Surround41);
    assert!(matches!(layout(0, true), Layout::Custom(_)));
    for amode in 0..10 {
        for lfe in [false, true] {
            let l = layout(amode, lfe);
            assert_eq!(l.channels(), AMODE_CHANNELS[amode as usize] + lfe as usize, "AMODE {amode}");
        }
    }
}

#[test]
fn sum_difference_pairs_follow_amode_channel_order() {
    assert_eq!(sum_diff_pairs(9), (Some((1, 2)), Some((3, 4))), "C L R SL SR");
    assert_eq!(sum_diff_pairs(2), (Some((0, 1)), None), "L R");
    assert_eq!(sum_diff_pairs(0), (None, None));
}

#[test]
fn a_refused_frame_still_reports_its_layout() {
    // The layout is mapped from the header before any audio is parsed, as
    // a caller reporting "dropped, with the reason" wants it.
    let mut d = Decoder::new();
    assert_eq!(d.layout(), None);
    assert!(d.decode(&frame_with_adpcm()).is_err());
    assert_eq!(d.layout(), Some(Layout::Surround51Side));
    let info = d.info().unwrap();
    assert_eq!((info.amode, info.lfe, info.sample_rate), (9, true, 48_000));
    assert!(!d.hf_vq_skipped());
}
