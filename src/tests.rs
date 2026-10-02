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

/// D.10.2 path: with a supplied code book, a VQ subband is `SCALES ×
/// element` for the subframe's samples (Table 5-29); without one it stays
/// silent and the decoder says so.
#[test]
fn hf_vq_subbands_use_a_supplied_book() {
    let entries: Vec<[i8; vq::HF_VQ_LEN]> =
        (0..vq::HF_VQ_VECTORS).map(|v| std::array::from_fn(|m| ((v * 3 + m * 5) % 200) as i8 - 100)).collect();
    let book = HfVqCodebook::from_entries(&entries).unwrap();
    // One channel: SUBS 4, VQSUB 2 → subbands 2 and 3 are VQ-coded.
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |v: u32, n: usize| {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    };
    push(2, 5); // SUBS = 4
    push(1, 5); // VQSUB = 2
    push(0, 3 + 2 + 3); // JOINX THUFF SHUFF
    push(5, 3); // BHUFF linear 4-bit
    push(0, 1 + 4 * 2 + 5 * 3); // SEL all Huffman …
    push(0, 2 * 10); // … so ten ADJ indices
    push(777, 10); // VQ index, subband 2
    push(5, 10); // VQ index, subband 3
    let bytes = pack(&bits, bits.len().div_ceil(8) as u32);
    let mut r = bits::BitReader::new(&bytes);
    let p = core::parse_coding_params(&mut r, 1, &[]).unwrap();
    assert_eq!((p.subs[0], p.vqsub[0]), (4, 2));
    let mut si = core::SideInfo {
        ssc: 2,
        pmode: [[false; 32]; 8],
        pvq: [[0; 32]; 8],
        abits: [[0; 32]; 8],
        tmode: [[0; 32]; 8],
        scales: [[[0.0; 2]; 32]; 8],
        join_scales: [[0.0; 32]; 8],
    };
    si.scales[0][2][0] = 1000.0;
    si.scales[0][3][0] = 10.0;
    let mut bufs = vec![core::ChannelBuf::new(32, 16)];
    let mut skipped = false;
    core::read_hf_vq(&mut r, &p, &si, 0, &mut bufs, Some(&book), &mut skipped).unwrap();
    assert!(!skipped);
    for m in 0..16 {
        assert_eq!(bufs[0].s[2][m], 1000.0 * entries[777][m] as f64 / 16.0);
        assert_eq!(bufs[0].s[3][m], 10.0 * entries[5][m] as f64 / 16.0);
    }
    let mut r = bits::BitReader::new(&bytes);
    let p = core::parse_coding_params(&mut r, 1, &[]).unwrap();
    let mut bufs = vec![core::ChannelBuf::new(32, 16)];
    core::read_hf_vq(&mut r, &p, &si, 0, &mut bufs, None, &mut skipped).unwrap();
    assert!(skipped);
    assert!(bufs[0].s.iter().flatten().all(|v| *v == 0.0));
}

/// D.9 is a linear-phase low-pass prototype once its printed sign changes
/// (every second block of 128 taps, 6.2.4.7) are undone, with unit DC gain.
#[test]
fn x96_prototype_is_linear_phase_with_unit_gain() {
    let g: Vec<f64> =
        tables::X96_QMF_FIR.iter().enumerate().map(|(n, &v)| if (n / 128) % 2 == 1 { -v } else { v }).collect();
    for n in 0..512 {
        assert_eq!(g[n], g[1023 - n], "tap {n}");
    }
    assert!((g.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    // The peak is at the centre.
    assert_eq!(g.iter().cloned().fold(0.0, f64::max), g[511]);
}

/// D.11: the DmixTable entries are 2^15 · 10^(dB/20) on the stated grid
/// (−60…−30 dB in 0.5 dB, −29.75…−15 in 0.25, −14.875…0 in 0.125), and
/// InvDmixTbl is 2^16 over the same gain — except that the "−3 dB" entry
/// is the exact 1/√2 (0.707107, as printed), the downmix gain of XCh.
#[test]
fn downmix_table_follows_its_grid() {
    for i in 0..241usize {
        let db = if i <= 60 {
            -60.0 + 0.5 * i as f64
        } else if i <= 120 {
            -30.0 + 0.25 * (i - 60) as f64
        } else {
            -15.0 + 0.125 * (i - 120) as f64
        };
        let gain = if db == -3.0 { std::f64::consts::FRAC_1_SQRT_2 } else { 10f64.powf(db / 20.0) };
        let want = 32768.0 * gain;
        assert!((tables::DMIX_TABLE[i] as f64 - want).abs() <= 1.0, "index {i}: {} vs {want:.1}", tables::DMIX_TABLE[i]);
        if i >= 40 {
            let inv = 65536.0 / gain;
            let got = tables::INV_DMIX_TABLE[i - 40] as f64;
            assert!((got - inv).abs() / inv < 1e-4, "inverse index {i}: {got} vs {inv:.1}");
        }
    }
}
