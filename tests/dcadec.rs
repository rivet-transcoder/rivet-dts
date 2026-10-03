//! This crate's decoder against an independent one, run as a black box:
//! libdca's `dcadec` command-line tool (VideoLAN; Debian / Ubuntu package
//! `libdca-utils`). Only its output is used — its source was not read.
//!
//! The streams are this crate's encoder's: a different known signal per
//! channel (tones, a multitone, noise, a chirp, each over a little noise so
//! the alignment is unambiguous), at 48, 44.1 and 32 kHz, at full and
//! reduced bit rates, with transient detection on and off. `dcadec -o wav6
//! -r` decodes each to 32-bit float WAV (no dynamic range compression),
//! this crate's decoder decodes the same bytes, and per channel — matched
//! by content, since the tool orders channels its own way — the two must
//! be sample-aligned, at the same level, and agree once the level is
//! divided out to a small relative RMS difference. That checks the parts
//! of the decoder an encoder written alongside it could share a mistake
//! with: the synthesis bank and its gain, the scale factor, step size and
//! LFE tables, the bit allocation and every entropy code the encoder uses.
//!
//! What the tool cannot check: `FILTS` = 1 streams (it does not decode
//! them; every public stream is `FILTS` = 0), and layouts its WAV output
//! does not carry one for one (see `compared_layouts`). Streams that
//! predict (ADPCM) are not sent either: this crate's encoder predicts only
//! with a code book the caller supplies, and the test book is not D.10.1.
//!
//! Without `dcadec` on PATH (or named by `DCADEC`) each test says so and
//! passes, unless `DTS_REQUIRE_DCADEC` is set (as in CI's job that installs
//! it), in which case a missing tool fails them.

use dts::{Decoder, Encoder, EncoderConfig, Layout, Speaker};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Command;

fn dcadec() -> Option<String> {
    let bin = std::env::var("DCADEC").unwrap_or_else(|_| "dcadec".to_string());
    let ok = Command::new(&bin).arg("-h").output().is_ok();
    assert!(
        ok || std::env::var_os("DTS_REQUIRE_DCADEC").is_none(),
        "DTS_REQUIRE_DCADEC is set but `{bin}` cannot be run"
    );
    if !ok {
        eprintln!("`{bin}` not found: skipping the comparison with libdca (set DCADEC or put dcadec on PATH)");
    }
    ok.then_some(bin)
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

/// A different signal per channel, so channels can be matched by content:
/// distinct tones, a multitone, noise, a chirp, a modulated tone; the LFE
/// gets low tones.
fn channel_signal(kind: usize, is_lfe: bool, rate: f64, n: usize) -> Vec<f64> {
    let t = |i: usize| i as f64 / rate;
    if is_lfe {
        return (0..n).map(|i| 0.4 * (2.0 * PI * 40.0 * t(i)).sin() + 0.2 * (2.0 * PI * 90.0 * t(i)).sin()).collect();
    }
    let f0 = 300.0 + 170.0 * kind as f64;
    // A little noise under every signal makes the alignment unambiguous.
    let floor = noise(100 + kind as u64, n);
    let tonal: Vec<f64> = match kind % 5 {
        0 => (0..n).map(|i| 0.5 * (2.0 * PI * f0 * t(i)).sin()).collect(),
        1 => (0..n)
            .map(|i| [220.0, 1000.0, 3150.0, 7000.0].iter().map(|f| 0.12 * (2.0 * PI * f * t(i)).sin()).sum())
            .collect(),
        2 => noise(kind as u64 + 7, n).iter().map(|v| 0.25 * v).collect(),
        3 => {
            let dur = n as f64 / rate;
            (0..n).map(|i| 0.3 * (2.0 * PI * (100.0 * t(i) + (14_900.0 / (2.0 * dur)) * t(i) * t(i))).sin()).collect()
        }
        _ => (0..n).map(|i| 0.4 * (2.0 * PI * f0 * 2.5 * t(i)).sin() * (2.0 * PI * 3.0 * t(i)).cos()).collect(),
    };
    tonal.iter().zip(&floor).map(|(a, b)| a + 0.05 * b).collect()
}

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

/// The stream as one byte string, and this crate's decode of it, per channel.
fn encode_and_decode(cfg: EncoderConfig, seconds: f64) -> (Vec<u8>, Vec<Vec<f64>>) {
    let layout = cfg.layout;
    let rate = cfg.sample_rate;
    let n = (rate as f64 * seconds) as usize;
    let speakers = layout.speakers();
    let input: Vec<Vec<f64>> =
        speakers.iter().enumerate().map(|(i, s)| channel_signal(i, *s == Speaker::LFE, rate as f64, n)).collect();
    let mut enc = Encoder::new(cfg).expect("valid configuration");
    let inter: Vec<f32> = (0..n).flat_map(|i| input.iter().map(move |c| c[i] as f32)).collect();
    let mut frames = enc.encode(&inter).unwrap();
    frames.extend(enc.flush().unwrap());
    let mut dec = Decoder::new();
    let mut out = vec![Vec::new(); speakers.len()];
    for f in &frames {
        for d in dec.decode(f).unwrap() {
            assert_eq!(d.layout, layout);
            for (k, s) in d.samples.iter().enumerate() {
                out[k % d.channels].push(*s as f64);
            }
        }
    }
    (frames.concat(), out)
}

/// A WAVE file's samples per channel: 16-bit PCM or 32-bit float, plain or
/// `WAVE_FORMAT_EXTENSIBLE`.
fn read_wav(b: &[u8]) -> (u32, Vec<Vec<f64>>) {
    if &b[..4] != b"RIFF" {
        // Raw interleaved f32 stereo (`-o float`).
        let mut out = vec![Vec::new(); 2];
        for (i, v) in b.as_chunks::<4>().0.iter().enumerate() {
            out[i % 2].push(f32::from_le_bytes(*v) as f64);
        }
        return (48_000, out);
    }
    assert!(b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE", "not a WAVE file");
    let u16le = |p: usize| u16::from_le_bytes([b[p], b[p + 1]]);
    let u32le = |p: usize| u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
    let (mut fmt, mut chans, mut rate, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut p = 12;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let len = u32le(p + 4) as usize;
        let body = p + 8;
        if id == b"fmt " {
            fmt = u16le(body);
            chans = u16le(body + 2) as usize;
            rate = u32le(body + 4);
            bits = u16le(body + 14);
            if fmt == 0xFFFE {
                fmt = u16le(body + 24); // the sub-format GUID's first two bytes
            }
        } else if id == b"data" {
            // Tools that write to a pipe leave the size unset: read to the end.
            let end = if len == 0 || body + len > b.len() { b.len() } else { body + len };
            let data = &b[body..end];
            let mut out = vec![Vec::new(); chans];
            match (fmt, bits) {
                (3, 32) => {
                    for (i, s) in data.as_chunks::<4>().0.iter().enumerate() {
                        out[i % chans].push(f32::from_le_bytes(*s) as f64);
                    }
                }
                (1, 16) => {
                    for (i, s) in data.as_chunks::<2>().0.iter().enumerate() {
                        out[i % chans].push(i16::from_le_bytes(*s) as f64 / 32768.0);
                    }
                }
                _ => panic!("WAVE format {fmt} at {bits} bits not handled"),
            }
            return (rate, out);
        }
        p = body + len + (len & 1);
    }
    panic!("no data chunk");
}

fn energy(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// Relative RMS of `theirs[i + lag]` against `ours[i]` over the middle of
/// the signal.
fn rel_rms(ours: &[f64], theirs: &[f64], lag: isize, gain: f64, edge: usize) -> f64 {
    let n = ours.len().min(theirs.len());
    let (mut e, mut s) = (0.0, 0.0);
    for (i, o) in ours.iter().enumerate().take(n.saturating_sub(edge)).skip(edge) {
        let j = i as isize + lag;
        if j < 0 || j as usize >= theirs.len() {
            continue;
        }
        e += (theirs[j as usize] / gain - o).powi(2);
        s += o * o;
    }
    (e / s.max(1e-30)).sqrt()
}

/// Least-squares gain of `theirs[i + lag]` over `ours[i]`.
fn gain(ours: &[f64], theirs: &[f64], lag: isize, edge: usize) -> f64 {
    let n = ours.len().min(theirs.len());
    let (mut x, mut s) = (0.0, 0.0);
    for (i, o) in ours.iter().enumerate().take(n.saturating_sub(edge)).skip(edge) {
        let j = i as isize + lag;
        if j >= 0 && (j as usize) < theirs.len() {
            x += theirs[j as usize] * o;
            s += o * o;
        }
    }
    x / s.max(1e-30)
}

/// One stream through both decoders: the worst relative RMS error over
/// the channels, after alignment and matching.
fn compare(bin: &str, cfg: EncoderConfig, dir: &Path) -> Result<Agreement, String> {
    let label = format!(
        "{} {} Hz {} b/s{}{}",
        cfg.layout.name(),
        cfg.sample_rate,
        cfg.bit_rate,
        if cfg.perfect_reconstruction { " FILTS=1" } else { "" },
        if cfg.transients { "" } else { " no-transients" }
    );
    let rate = cfg.sample_rate;
    let speakers = cfg.layout.speakers().to_vec();
    let (stream, ours) = encode_and_decode(cfg, 1.0);
    let path = dir.join(format!("{}.dts", label.replace(|c: char| !c.is_ascii_alphanumeric(), "_")));
    std::fs::write(&path, &stream).unwrap();
    let out = Command::new(bin).args(["-o", "wav6", "-r"]).arg(&path).output().expect("run dcadec");
    assert!(out.status.success(), "{label}: dcadec failed: {}", String::from_utf8_lossy(&out.stderr));
    let (their_rate, theirs) = read_wav(&out.stdout);
    let _ = std::fs::remove_file(&path);
    if their_rate != rate {
        return Err(format!("{label}: dcadec's output is at {their_rate} Hz"));
    }
    // Their channels that carry anything.
    let live: Vec<usize> = (0..theirs.len()).filter(|&c| energy(&theirs[c]) > 1e-9).collect();
    if live.len() != speakers.len() {
        return Err(format!("{label}: dcadec output {} live channels of {}", live.len(), theirs.len()));
    }

    // The delay between the decoders, from the first non-LFE channel.
    let c0 = speakers.iter().position(|s| *s != Speaker::LFE).unwrap();
    let mut lag = (0isize, f64::INFINITY);
    for &t in &live {
        for l in -1024isize..=1024 {
            let r = rel_rms(&ours[c0][..ours[c0].len().min(24576)], &theirs[t], l, 1.0, 8192);
            if r < lag.1 {
                lag = (l, r);
            }
        }
    }
    let lag = lag.0;
    let mut used = Vec::new();
    let mut agreement = Agreement { gain: 0.0, shape: 0.0 };
    let mut line = String::new();
    for (c, s) in speakers.iter().enumerate() {
        let (t, r) = live
            .iter()
            .map(|&t| (t, rel_rms(&ours[c], &theirs[t], lag, 1.0, 2048)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if used.contains(&t) {
            return Err(format!("{label}: two of our channels match dcadec's channel {t}"));
        }
        used.push(t);
        // The least-squares gain of theirs over ours, and what is left
        // once it is divided out.
        let g = gain(&ours[c], &theirs[t], lag, 2048);
        let shape = rel_rms(&ours[c], &theirs[t], lag, g, 2048);
        line += &format!(" {s:?}→{t}: gain {g:.6} rel {r:.2e} (after gain {shape:.2e})");
        agreement.gain = agreement.gain.max((g - 1.0).abs());
        agreement.shape = agreement.shape.max(shape);
    }
    eprintln!("{label}: lag {lag},{line}");
    if lag != 0 {
        return Err(format!("{label}: the decoders are {lag} samples apart"));
    }
    Ok(agreement)
}

/// How far the two decoders' outputs are apart, over the channels: the
/// largest deviation of the gain from 1, and the largest relative RMS
/// difference once the gain is divided out.
struct Agreement {
    gain: f64,
    shape: f64,
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dts-dcadec-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Once the gain is divided out, the relative RMS difference allowed per
/// channel. Both decoders are floating-point renderings of the same
/// arithmetic.
const SHAPE_TOLERANCE: f64 = 1e-4;

/// The gain deviation allowed per channel. At the full rates, where one or
/// two channels get the finest quantisers, the two decoders' levels differ
/// by up to 0.14 % (mono) — consistent with D.2.1's step sizes, which the
/// specification prints as integers × 2^-22 (21, 42, 84, … for `ABITS` 26,
/// 25, 24, …) and so only to a part in a thousand; this decoder uses the
/// printed values. Everywhere else they agree to a few parts per million.
const GAIN_TOLERANCE_FULL_RATE: f64 = 2e-3;
const GAIN_TOLERANCE: f64 = 2e-5;

/// The configuration with the QMF prototype the tool decodes. libdca
/// decodes `FILTS` = 0 streams (the non-perfect prototype, the one every
/// public stream uses) but not `FILTS` = 1 ones, so the comparison is on
/// `FILTS` = 0; the perfect-reconstruction bank is checked by the round
/// trips and the spec-derived unit tests only.
fn config(rate: u32, layout: Layout, bit_rate: u32) -> EncoderConfig {
    let mut cfg = EncoderConfig::new(rate, layout, bit_rate);
    cfg.perfect_reconstruction = false;
    cfg
}

/// The layouts the tool's multichannel WAV output carries one for one:
/// mono, and every core arrangement of three or more channels with the
/// LFE. Its two-channel output is 16-bit PCM that clips (for the public
/// stereo streams as well), and for three or more channels without the LFE
/// it writes channels that are mixtures, or an extra one — so those
/// layouts, decoded by this crate alone, are covered by the round trips.
fn compared_layouts() -> Vec<Layout> {
    let mut v = vec![Layout::Mono];
    v.extend(all_layouts().into_iter().filter(|l| l.speakers().contains(&Speaker::LFE) && l.channels() >= 3));
    v
}

fn check(bin: &str, cfgs: Vec<EncoderConfig>, gain_tolerance: f64, dir: &Path) {
    let (mut gain, mut shape) = (0.0f64, 0.0f64);
    let n = cfgs.len();
    for cfg in cfgs {
        let a = compare(bin, cfg, dir).unwrap_or_else(|e| panic!("{e}"));
        gain = gain.max(a.gain);
        shape = shape.max(a.shape);
    }
    eprintln!("{n} streams: worst gain deviation {gain:.2e}, worst relative RMS after gain {shape:.2e}");
    assert!(shape < SHAPE_TOLERANCE, "relative RMS {shape:.2e} ≥ {SHAPE_TOLERANCE:e}");
    assert!(gain < gain_tolerance, "gain deviation {gain:.2e} ≥ {gain_tolerance:e}");
}

#[test]
fn every_compared_layout_agrees_with_libdca_at_full_rate() {
    let Some(bin) = dcadec() else { return };
    let dir = scratch("full");
    let mut cfgs = Vec::new();
    for (rate, bit_rate) in [(48_000u32, 1_536_000u32), (44_100, 1_411_200), (32_000, 1_024_000)] {
        cfgs.extend(compared_layouts().into_iter().map(|l| config(rate, l, bit_rate)));
    }
    check(&bin, cfgs, GAIN_TOLERANCE_FULL_RATE, &dir);
}

#[test]
fn every_compared_layout_agrees_with_libdca_at_lower_rates() {
    let Some(bin) = dcadec() else { return };
    let dir = scratch("lower");
    let mut cfgs = Vec::new();
    for (rate, bit_rate) in [(48_000u32, 384_000u32), (44_100, 256_000)] {
        cfgs.extend(compared_layouts().into_iter().map(|l| config(rate, l, bit_rate)));
    }
    for (layout, rate, bit_rate) in [
        (Layout::Surround51Side, 48_000, 768_000),
        (Layout::Surround51Side, 48_000, 448_000),
        (Layout::Stereo21, 32_000, 192_000),
        (Layout::Mono, 48_000, 128_000),
    ] {
        cfgs.push(config(rate, layout, bit_rate));
    }
    // Transient detection off: one scale factor per subband throughout.
    for (rate, bit_rate) in [(48_000u32, 768_000u32), (44_100, 384_000)] {
        let mut cfg = config(rate, Layout::Surround51Side, bit_rate);
        cfg.transients = false;
        cfgs.push(cfg);
    }
    check(&bin, cfgs, GAIN_TOLERANCE, &dir);
}
