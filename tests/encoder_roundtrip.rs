//! Round trips through this crate's encoder and decoder: no other
//! implementation is involved. For every core layout, at 48 and 44.1 kHz
//! and several Table 5-7 bit rates, known signals are encoded, every frame
//! is checked (sync word, `FSIZE`, the constant size rate control
//! promises), the stream is decoded, and the output — aligned by the
//! encoder's stated delay — is compared with the input: SNR per channel,
//! and the gain of a multitone at frequencies from 50 Hz up.

use dts::{AdpcmCodebook, AdpcmFallback, Decoder, Encoder, EncoderConfig, Layout, Speaker};
use std::f64::consts::PI;
use std::sync::Arc;

/// Every core arrangement (AMODE 0, 2, 5–9), with and without the LFE.
fn all_layouts() -> Vec<Layout> {
    use Speaker::*;
    let cores: [&[Speaker]; 7] =
        [&[FC], &[FL, FR], &[FC, FL, FR], &[FL, FR, BC], &[FC, FL, FR, BC], &[FL, FR, SL, SR], &[FC, FL, FR, SL, SR]];
    let mut v = Vec::new();
    for c in cores {
        v.push(Layout::from_speakers(c));
        let mut l = c.to_vec();
        l.push(LFE);
        v.push(Layout::from_speakers(&l));
    }
    v
}

/// Deterministic white noise in [-1, 1).
fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// A different test signal per channel: sines, a multitone, noise, a
/// chirp; the LFE gets low tones.
fn channel_signal(kind: usize, is_lfe: bool, rate: f64, n: usize) -> Vec<f64> {
    let t = |i: usize| i as f64 / rate;
    if is_lfe {
        return (0..n).map(|i| 0.4 * (2.0 * PI * 40.0 * t(i)).sin() + 0.2 * (2.0 * PI * 90.0 * t(i)).sin()).collect();
    }
    match kind % 5 {
        0 => (0..n).map(|i| 0.5 * (2.0 * PI * 440.0 * t(i)).sin()).collect(),
        1 => (0..n)
            .map(|i| [220.0, 1000.0, 3150.0, 7000.0].iter().map(|f| 0.12 * (2.0 * PI * f * t(i)).sin()).sum())
            .collect(),
        2 => noise(kind as u64, n).iter().map(|v| 0.25 * v).collect(),
        3 => {
            // Linear chirp 100 Hz → 15 kHz over the signal.
            let dur = n as f64 / rate;
            (0..n).map(|i| 0.3 * (2.0 * PI * (100.0 * t(i) + (14_900.0 / (2.0 * dur)) * t(i) * t(i))).sin()).collect()
        }
        _ => (0..n).map(|i| 0.4 * (2.0 * PI * 1234.5 * t(i)).sin() * (2.0 * PI * 3.0 * t(i)).cos()).collect(),
    }
}

struct Encoded {
    frames: Vec<Vec<u8>>,
    delay: usize,
    frame_bytes: usize,
}

fn encode(cfg: EncoderConfig, input: &[Vec<f64>]) -> Encoded {
    let channels = input.len();
    let n = input[0].len();
    let mut enc = Encoder::new(cfg).expect("valid configuration");
    let mut frames = Vec::new();
    // Feed in uneven chunks, as a caller would.
    let mut pos = 0;
    let mut chunk = 700;
    while pos < n {
        let end = (pos + chunk).min(n);
        let inter: Vec<f32> = (pos..end).flat_map(|i| (0..channels).map(move |c| input[c][i] as f32)).collect();
        frames.extend(enc.encode(&inter).unwrap());
        pos = end;
        chunk = chunk * 7 % 1500 + 1;
    }
    frames.extend(enc.flush().unwrap());
    Encoded { frames, delay: enc.delay(), frame_bytes: enc.frame_bytes() }
}

/// Decode and return per-channel PCM, checking every frame on the way.
fn decode(e: &Encoded, layout: Layout, rate: u32) -> Vec<Vec<f64>> {
    decode_with(Decoder::new(), e, layout, rate)
}

fn decode_with(mut dec: Decoder, e: &Encoded, layout: Layout, rate: u32) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new(); layout.channels()];
    for (i, f) in e.frames.iter().enumerate() {
        assert_eq!(f.len(), e.frame_bytes, "frame {i}: rate control must hold the frame size");
        assert_eq!(&f[..4], &[0x7F, 0xFE, 0x80, 0x01], "frame {i}: sync");
        assert_eq!(dts::frame_len(f).unwrap(), f.len(), "frame {i}: FSIZE + 1");
        for d in dec.decode(f).unwrap_or_else(|err| panic!("frame {i}: {err}")) {
            assert_eq!(d.layout, layout);
            assert_eq!(d.sample_rate, rate);
            for (k, s) in d.samples.iter().enumerate() {
                out[k % d.channels].push(*s as f64);
            }
        }
    }
    out
}

/// SNR (dB) of `out` against `inp` delayed by `delay`, skipping the edges.
fn snr(inp: &[f64], out: &[f64], delay: usize) -> f64 {
    let skip = 2048;
    let (mut s, mut e) = (0.0, 0.0);
    for i in skip..inp.len() - skip {
        s += inp[i] * inp[i];
        e += (out[i + delay] - inp[i]).powi(2);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

struct Case {
    layout: Layout,
    rate: u32,
    bit_rate: u32,
    min_snr: f64,
}

fn run(case: &Case, codebook: Option<Arc<AdpcmCodebook>>) -> Vec<f64> {
    let n = case.rate as usize; // one second
    let speakers = case.layout.speakers();
    let input: Vec<Vec<f64>> = speakers
        .iter()
        .enumerate()
        .map(|(i, s)| channel_signal(i, *s == dts::Speaker::LFE, case.rate as f64, n))
        .collect();
    let mut cfg = EncoderConfig::new(case.rate, case.layout, case.bit_rate);
    cfg.adpcm_codebook = codebook;
    let e = encode(cfg, &input);
    let out = decode(&e, case.layout, case.rate);
    assert!(out[0].len() >= n + e.delay, "decoded stream covers the input");
    let snrs: Vec<f64> = (0..speakers.len()).map(|c| snr(&input[c], &out[c], e.delay)).collect();
    eprintln!(
        "{:>10} {:>5} Hz {:>8} b/s: SNR per channel {}",
        case.layout.name(),
        case.rate,
        case.bit_rate,
        speakers
            .iter()
            .zip(&snrs)
            .map(|(s, v)| format!("{s:?} {v:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (s, v) in speakers.iter().zip(&snrs) {
        let floor = if *s == dts::Speaker::LFE { 30.0 } else { case.min_snr };
        assert!(*v >= floor, "{} {} Hz {} b/s: {s:?} SNR {v:.1} dB < {floor}", case.layout.name(), case.rate, case.bit_rate);
    }
    snrs
}

#[test]
fn every_layout_round_trips_at_48k_full_rate() {
    for layout in all_layouts() {
        run(&Case { layout, rate: 48_000, bit_rate: 1_536_000, min_snr: 45.0 }, None);
    }
}

#[test]
fn every_layout_round_trips_at_44k1() {
    for layout in all_layouts() {
        run(&Case { layout, rate: 44_100, bit_rate: 1_411_200, min_snr: 45.0 }, None);
    }
}

#[test]
fn lower_rates_still_round_trip() {
    for (layout, rate, bit_rate, min_snr) in [
        (Layout::Surround51Side, 48_000, 768_000, 25.0),
        (Layout::Stereo, 48_000, 384_000, 25.0),
        (Layout::Stereo, 44_100, 256_000, 15.0),
        (Layout::Stereo, 32_000, 192_000, 15.0),
        (Layout::Mono, 48_000, 128_000, 15.0),
    ] {
        run(&Case { layout, rate, bit_rate, min_snr }, None);
    }
}

/// Goertzel magnitude of `x` at `f`.
fn tone_level(x: &[f64], f: f64, rate: f64) -> f64 {
    let w = 2.0 * PI * f / rate;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() {
        re += v * (w * i as f64).cos();
        im -= v * (w * i as f64).sin();
    }
    (re * re + im * im).sqrt()
}

/// Gain of a 26-tone multitone from 50 Hz to 20 kHz (48 kHz) through the
/// round trip: within ±0.5 dB up to 18 kHz at full rate. The LFE's
/// passband is checked the same way below 120 Hz.
#[test]
fn frequency_response_is_flat() {
    for (rate, bit_rate) in [(48_000u32, 1_536_000u32), (44_100, 1_411_200)] {
        let r = rate as f64;
        let n = rate as usize;
        let freqs: Vec<f64> = (0..26).map(|i| 50.0 * (400.0f64).powf(i as f64 / 25.0)).filter(|f| *f < 0.45 * r).collect();
        let multitone: Vec<f64> = (0..n)
            .map(|i| freqs.iter().enumerate().map(|(k, f)| 0.03 * (2.0 * PI * f * i as f64 / r + k as f64).sin()).sum())
            .collect();
        let lfe_freqs = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0];
        let lfe: Vec<f64> = (0..n)
            .map(|i| lfe_freqs.iter().map(|f| 0.1 * (2.0 * PI * f * i as f64 / r).sin()).sum())
            .collect();
        let input = vec![multitone.clone(), multitone.clone(), lfe.clone()];
        let e = encode(EncoderConfig::new(rate, Layout::Stereo21, bit_rate), &input);
        let out = decode(&e, Layout::Stereo21, rate);
        let win = 4096..n - 4096;
        let mut worst = 0.0f64;
        let mut line = String::new();
        for f in &freqs {
            let a = tone_level(&input[0][win.clone()], *f, r);
            let b = tone_level(&out[0][win.start + e.delay..win.end + e.delay], *f, r);
            let db = 20.0 * (b / a).log10();
            line += &format!(" {:.0}:{db:+.2}", f);
            if *f <= 18_000.0 {
                worst = worst.max(db.abs());
            }
        }
        eprintln!("{rate} Hz {bit_rate} b/s FL gain (Hz:dB):{line}");
        assert!(worst <= 0.5, "{rate} Hz: passband deviation {worst:.2} dB");
        let mut lline = String::new();
        for f in lfe_freqs {
            let a = tone_level(&input[2][win.clone()], f, r);
            let b = tone_level(&out[2][win.start + e.delay..win.end + e.delay], f, r);
            let db = 20.0 * (b / a).log10();
            lline += &format!(" {f:.0}:{db:+.2}");
            assert!(db.abs() <= 1.0, "{rate} Hz: LFE gain at {f} Hz is {db:+.2} dB");
        }
        eprintln!("{rate} Hz LFE gain (Hz:dB):{lline}");
    }
}

/// With a prediction code book the encoder predicts tonal subbands and the
/// decoder, given the same book, follows exactly. Without the book the
/// decoder refuses those frames by name; with the estimate fallback it
/// decodes them, as concealment only.
#[test]
fn adpcm_round_trips_with_a_shared_code_book() {
    let book = Arc::new(AdpcmCodebook::private_test_book());
    let rate = 48_000u32;
    let n = rate as usize;
    let input: Vec<Vec<f64>> = (0..2)
        .map(|c| (0..n).map(|i| 0.4 * (2.0 * PI * (330.0 + 500.0 * c as f64) * i as f64 / rate as f64).sin()).collect())
        .collect();
    for bit_rate in [384_000u32, 256_000] {
        let mut cfg = EncoderConfig::new(rate, Layout::Stereo, bit_rate);
        cfg.adpcm_codebook = Some(book.clone());
        let e = encode(cfg, &input);
        let predicted = e.frames.iter().filter(|f| has_pmode(f)).count();
        eprintln!("ADPCM {bit_rate} b/s: {predicted} of {} frames predict at least one subband", e.frames.len());
        assert!(predicted > e.frames.len() / 2, "a steady tone should be predicted");

        // The same book on both sides: an exact decode.
        let mut dec = Decoder::new();
        dec.set_adpcm_codebook(Some(book.clone()));
        let out = decode_with(dec, &e, Layout::Stereo, rate);
        let with: Vec<f64> = (0..2).map(|c| snr(&input[c], &out[c], e.delay)).collect();
        // The same signal without prediction, for comparison.
        let plain = encode(EncoderConfig::new(rate, Layout::Stereo, bit_rate), &input);
        let out_plain = decode(&plain, Layout::Stereo, rate);
        let without: Vec<f64> = (0..2).map(|c| snr(&input[c], &out_plain[c], plain.delay)).collect();
        // No book: refused by name.
        let mut dec = Decoder::new();
        let refused = e.frames.iter().filter(|f| matches!(dec.decode(f), Err(dts::Error::Unsupported(w)) if w.contains("D.10.1"))).count();
        // (has_pmode looks at the first channel only.)
        assert!(refused >= predicted, "every predicted frame is refused without the book");
        // No book, estimate fallback: decodes, approximately.
        let mut dec = Decoder::new();
        dec.set_adpcm_fallback(AdpcmFallback::Estimate);
        let mut est = vec![Vec::new(); 2];
        for f in &e.frames {
            for d in dec.decode(f).unwrap() {
                for (k, s) in d.samples.iter().enumerate() {
                    est[k % 2].push(*s as f64);
                }
            }
        }
        let estimated: Vec<f64> = (0..2).map(|c| snr(&input[c], &est[c], e.delay)).collect();
        eprintln!(
            "ADPCM {bit_rate} b/s SNR: predicted+book {:.1}/{:.1} dB, unpredicted {:.1}/{:.1} dB,              predicted+estimate {:.1}/{:.1} dB ({} subband predictors estimated)",
            with[0], with[1], without[0], without[1], estimated[0], estimated[1], dec.adpcm_estimated()
        );
        for c in 0..2 {
            assert!(with[c] > 40.0, "decoded with the book: {:.1} dB", with[c]);
            // The estimate is concealment (see AdpcmFallback::Estimate):
            // it must decode every frame without blowing up, nothing more.
            assert!(estimated[c] > -3.0, "estimated predictor rang: {:.1} dB", estimated[c]);
            assert!(est[c].iter().all(|v| v.abs() < 2.0), "estimated output stays bounded");
        }
    }
}

#[allow(clippy::needless_range_loop)] // field order mirrors Table 5-21
/// Whether the frame's first channel has any `PMODE` set (parses only as
/// far as the PMODE flags of a one-subframe frame with the encoder's
/// header layout).
fn has_pmode(f: &[u8]) -> bool {
    let bit = |p: usize| (f[p >> 3] >> (7 - (p & 7))) & 1;
    let bits = |p: usize, n: usize| (0..n).fold(0u32, |a, i| (a << 1) | bit(p + i) as u32);
    let pchs = bits(104 + 4, 3) as usize + 1;
    let subs: Vec<usize> = (0..pchs).map(|c| bits(111 + 5 * c, 5) as usize + 2).collect();
    let mut p = 111 + pchs * (5 + 5 + 3 + 2 + 3 + 3);
    // SEL fields.
    let mut sel = vec![[0u32; 10]; pchs];
    for c in 0..pchs {
        sel[c][0] = bits(p, 1);
        p += 1;
    }
    for n in 1..5 {
        for c in 0..pchs {
            sel[c][n] = bits(p, 2);
            p += 2;
        }
    }
    for n in 5..10 {
        for c in 0..pchs {
            sel[c][n] = bits(p, 3);
            p += 3;
        }
    }
    for n in 0..10 {
        for s in &sel {
            let huff = match n {
                0 => s[0] == 0,
                1..=4 => s[n] < 3,
                _ => s[n] < 7,
            };
            if huff {
                p += 2;
            }
        }
    }
    p += 5; // SSC PSC
    (0..subs[0]).any(|k| bit(p + k) == 1)
}

/// Unused-input sanity: a silent input encodes to frames that decode to
/// silence.
#[test]
fn silence_stays_silent() {
    let input = vec![vec![0.0; 10_000]; 6];
    let e = encode(EncoderConfig::new(48_000, Layout::Surround51Side, 1_536_000), &input);
    let out = decode(&e, Layout::Surround51Side, 48_000);
    assert!(out.iter().flatten().all(|v| *v == 0.0));
}
