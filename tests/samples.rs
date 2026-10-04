//! Public DTS streams, decoded with no reference output (none is published
//! with them): every packet must decode or be refused for the one reason
//! this decoder refuses valid streams (ADPCM prediction without the D.10.1
//! code book); with the estimate fallback every packet must decode, to the
//! layout and rate the stream declares, at plausible levels; and where a
//! stream carries an extension, the extension must be consistent with the
//! core it extends (XCh: the front channels untouched and the surrounds
//! changed by exactly the −3 dB back-centre downmix; XBR: a small
//! correction; X96: the core's spectrum below 24 kHz, plus content above).
//!
//! The streams are not in this repository. They are DTS's and others'
//! demonstration streams as collected at <https://streams.videolan.org/samples/A-codecs/DTS/>
//! (a public sample archive; only the files are used, as data). Put them in
//! a directory and point `DTS_SAMPLES_DIR` at it (CI's samples job does;
//! `tools/fetch_samples.sh` downloads them); each one is checked against
//! the SHA-256 below before use. Without the variable these tests say so
//! and pass, unless `DTS_REQUIRE_SAMPLES` is set.

use std::path::PathBuf;

use dts::{AdpcmFallback, Decoder, Extensions, Frame};

/// One sample stream: file name, URL path below the archive root, SHA-256,
/// expected layout name and output rate, extensions that must be decoded.
struct Sample {
    file: &'static str,
    sha256: &'static str,
    layout: &'static str,
    rate: u32,
    decoded: Extensions,
    /// Extensions present but not decoded (XLL).
    skipped_xll: bool,
}

const fn ext(xch: bool, xxch: bool, x96: bool, xbr: bool, exss: bool) -> Extensions {
    Extensions {
        xch,
        xxch,
        x96,
        xbr,
        exss,
        xll: false,
        lbr: false,
    }
}

const NONE: Extensions = ext(false, false, false, false, false);

/// `(archive path, …)`. URLs: `https://streams.videolan.org/samples/A-codecs/DTS/` + path.
const SAMPLES: &[(&str, Sample)] = &[
    (
        "dts/3-1.dts",
        Sample {
            file: "3-1.dts",
            sha256: "c21f1dd9a21f96ff198365b22edff20d0c35d8c8c659dd27aac34b720cfb7b9a",
            layout: "4.0",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/5.1 24bit.dts",
        Sample {
            file: "5.1 24bit.dts",
            sha256: "3956047bd9706aa373e3d8b7c8994843374b67351d670ffbb2cd480c3f3190d6",
            layout: "5.1(side)",
            rate: 48_000,
            decoded: NONE,
            skipped_xll: false,
        },
    ),
    (
        "dts/96-24.dts",
        Sample {
            file: "96-24.dts",
            sha256: "422c8db3496708dbedc67a97b8d8d2652f75e85516d7b46a58ef2df00d48df7b",
            layout: "5.1(side)",
            rate: 96_000,
            decoded: ext(false, false, true, false, false),
            skipped_xll: false,
        },
    ),
    (
        "dts/ES 6.1 - 5.1 16bit.dts",
        Sample {
            file: "ES 6.1 - 5.1 16bit.dts",
            sha256: "57469d05109a39bdc48fc3af77ad873822844d4a90686e58e794f1e2237fc2d4",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(true, false, false, false, false),
            skipped_xll: false,
        },
    ),
    (
        "dts/ES 6.1 16bit.dts",
        Sample {
            file: "ES 6.1 16bit.dts",
            sha256: "a04708ac58c70b0da9c0bf63b24c74c4039a398eb2916de715019b1e1f809452",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(true, false, false, false, false),
            skipped_xll: false,
        },
    ),
    (
        "dts/ES 6.1 24bit.dts",
        Sample {
            file: "ES 6.1 24bit.dts",
            sha256: "c4017d9426d5e9dae06a3ca681cd11526e438f499a20b5f12a5c4b418d1324f3",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(true, false, false, false, false),
            skipped_xll: false,
        },
    ),
    (
        "dts/Hi-Res 5.1 24bit.dts",
        Sample {
            file: "Hi-Res 5.1 24bit.dts",
            sha256: "300cba0e3f2d971921678aa08d19788301d4fa97784b0796e186e554e5998d66",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(false, true, false, true, true),
            skipped_xll: false,
        },
    ),
    (
        "dts/Hi-Res 6.1 24bit.dts",
        Sample {
            file: "Hi-Res 6.1 24bit.dts",
            sha256: "f182cda9e008073d9f7ccf77257ff8d05665397916c0dbbe39427ca68a7058bb",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(true, false, false, true, true),
            skipped_xll: false,
        },
    ),
    (
        "dts/Master Audio 2.0 16bit.dts",
        Sample {
            file: "Master Audio 2.0 16bit.dts",
            sha256: "34845219924fedc4c633a97c614f464f25011857d11b6ad1919e0c02f1abb3ce",
            layout: "stereo",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/Master Audio 5.0 96khz.dts",
        Sample {
            file: "Master Audio 5.0 96khz.dts",
            sha256: "3702d95a38cba3414968724e7bacb13440b81e79c6be753dc53c2a35813cdb39",
            layout: "5.0(side)",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/Master Audio 5.1 16bit.dts",
        Sample {
            file: "Master Audio 5.1 16bit.dts",
            sha256: "70418af672befaa22b192798f54b864eb74eb5621fdf2a71ca358cffb114e1b0",
            layout: "5.1(side)",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/Master Audio 7.1 24bit.dts",
        Sample {
            file: "Master Audio 7.1 24bit.dts",
            sha256: "0da506ccc59fdef1744bdbe178637199cad05207601e96fbf8d163694fe39c7e",
            layout: "5.1(side)",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/Master Audio 7.1.dts",
        Sample {
            file: "Master Audio 7.1.dts",
            sha256: "08b6289cceedadfd2e9e7be833e60bb8afd751564cf5e6543fecbc6a3a22f8d9",
            layout: "5.1(side)",
            rate: 48_000,
            decoded: ext(false, false, false, false, true),
            skipped_xll: true,
        },
    ),
    (
        "dts/dtswavsample14.wav",
        Sample {
            file: "dtswavsample14.wav",
            sha256: "f6a4889064e9f25eb873d502de8ae388d4a0b5f3776dbb5779d393f2179480e8",
            layout: "5.1(side)",
            rate: 44_100,
            decoded: NONE,
            skipped_xll: false,
        },
    ),
    (
        "dts/open bitrate.dts",
        Sample {
            file: "open bitrate.dts",
            sha256: "7069220f675bd6608d37190ddf0d2de1ece570eb057da84131e2bc9a5984d720",
            layout: "stereo",
            rate: 48_000,
            decoded: NONE,
            skipped_xll: false,
        },
    ),
    (
        "dts/padded.dts",
        Sample {
            file: "padded.dts",
            sha256: "b017346b8f09e3a27a599445d7367879dd2802542bd75b4ce59f0c5e3a5a9c12",
            layout: "5.1(side)",
            rate: 48_000,
            decoded: NONE,
            skipped_xll: false,
        },
    ),
    (
        "lotr_5.1_768.dts",
        Sample {
            file: "lotr_5.1_768.dts",
            sha256: "6c70137c8d4383668c034bd4993ba7e9cb10044a2165a35fe78bf2316388d8d8",
            layout: "6.1",
            rate: 48_000,
            decoded: ext(true, false, false, false, false),
            skipped_xll: false,
        },
    ),
    (
        "scissorhands-4.0-48_24.dts",
        Sample {
            file: "scissorhands-4.0-48_24.dts",
            sha256: "46c6d087c5e33ca6263c87c2aa752f6167db04f2292075a0831ff9d2264c3150",
            layout: "4.0",
            rate: 48_000,
            decoded: NONE,
            skipped_xll: false,
        },
    ),
];

fn samples_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("DTS_SAMPLES_DIR").map(PathBuf::from);
    assert!(
        dir.is_some() || std::env::var_os("DTS_REQUIRE_SAMPLES").is_none(),
        "DTS_REQUIRE_SAMPLES is set but DTS_SAMPLES_DIR is not"
    );
    dir
}

/// The stream, re-framed to 16-bit big-endian, after checking its SHA-256.
fn load(s: &Sample) -> Option<Vec<u8>> {
    let path = samples_dir()?.join(s.file);
    let raw = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var_os("DTS_REQUIRE_SAMPLES").is_none(),
                "{}: {e}",
                path.display()
            );
            eprintln!("{}: not present, skipped", s.file);
            return None;
        }
    };
    assert_eq!(hex(&sha256(&raw)), s.sha256, "{}: SHA-256", s.file);
    Some(
        dts::normalize_framing(&raw)
            .expect("a DTS stream")
            .into_owned(),
    )
}

/// Split into packets and decode each; `Err` per packet kept.
fn decode_all(bytes: &[u8], dec: &mut Decoder) -> Vec<Result<Vec<Frame>, dts::Error>> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= bytes.len() {
        let len = match dts::packet_len(&bytes[off..]) {
            Ok(l) => l,
            Err(e) => {
                out.push(Err(e));
                break;
            }
        };
        let end = (off + len).min(bytes.len());
        out.push(dec.decode(&bytes[off..end]));
        off = end;
    }
    out
}

/// The last packet of a cut-down sample may be cut too: a truncation
/// there is the file, not the decoder.
fn trailing_truncation(results: &[Result<Vec<Frame>, dts::Error>], i: usize) -> bool {
    i + 1 == results.len() && matches!(results[i], Err(dts::Error::Truncated { .. }))
}

/// Per output channel, concatenated.
fn planar(frames: &[Frame]) -> Vec<Vec<f32>> {
    let ch = frames.first().map_or(0, |f| f.channels);
    let mut out = vec![Vec::new(); ch];
    for f in frames {
        for (i, v) in f.samples.iter().enumerate() {
            out[i % ch].push(*v);
        }
    }
    out
}

fn decoder(fallback: AdpcmFallback, ext: Extensions) -> Decoder {
    let mut d = Decoder::new();
    d.set_adpcm_fallback(fallback);
    d.set_extensions(ext);
    d
}

#[test]
fn public_streams_decode_or_are_refused_only_for_adpcm() {
    for (_, s) in SAMPLES {
        let Some(bytes) = load(s) else { continue };
        let results = decode_all(&bytes, &mut Decoder::new());
        let (mut ok, mut adpcm) = (0, 0);
        for (i, r) in results.iter().enumerate() {
            match r {
                Ok(_) => ok += 1,
                Err(dts::Error::Unsupported(w)) if w.contains("D.10.1") => adpcm += 1,
                Err(_) if trailing_truncation(&results, i) => {}
                Err(e) => panic!("{}: packet {i}: {e}", s.file),
            }
        }
        eprintln!(
            "{:>28}: {} packets: {ok} decoded exactly, {adpcm} refused for ADPCM ({:.0} %)",
            s.file,
            results.len(),
            100.0 * adpcm as f64 / results.len() as f64
        );
    }
}

#[test]
fn public_streams_decode_completely_with_the_estimate() {
    for (_, s) in SAMPLES {
        let Some(bytes) = load(s) else { continue };
        let mut dec = decoder(AdpcmFallback::Estimate, Extensions::ALL);
        let results = decode_all(&bytes, &mut dec);
        let mut frames = Vec::new();
        for (i, r) in results.iter().enumerate() {
            match r {
                Ok(f) => frames.extend(f.iter().cloned()),
                Err(_) if trailing_truncation(&results, i) => {}
                Err(e) => panic!("{}: packet {i}: {e}", s.file),
            }
        }
        let info = dec.info().expect("frames decoded");
        assert_eq!(info.layout.name(), s.layout, "{}: layout", s.file);
        assert_eq!(info.sample_rate, s.rate, "{}: output rate", s.file);
        let want = s.decoded;
        let got = info.extensions;
        assert!(
            (!want.xch || got.xch)
                && (!want.xxch || got.xxch)
                && (!want.x96 || got.x96)
                && (!want.xbr || got.xbr)
                && (!want.exss || got.exss),
            "{}: decoded extensions {got:?}, want {want:?}",
            s.file
        );
        assert_eq!(info.skipped.xll, s.skipped_xll, "{}: XLL present", s.file);
        let pcm = planar(&frames);
        let mut line = String::new();
        for (c, spk) in info.layout.speakers().iter().enumerate() {
            let x = &pcm[c];
            assert!(
                x.iter().all(|v| v.is_finite()),
                "{}: non-finite output",
                s.file
            );
            let peak = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
            assert!(peak <= 1.0, "{}: {spk:?} peaks at {peak}", s.file);
            line += &format!(
                " {spk:?} {:.1}/{:.1}",
                20.0 * rms.max(1e-12).log10(),
                20.0 * (peak as f64).max(1e-12).log10()
            );
        }
        let total_rms = (pcm
            .iter()
            .flatten()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / pcm.iter().map(Vec::len).sum::<usize>() as f64)
            .sqrt();
        assert!(total_rms > 1e-4, "{}: decodes to near-silence", s.file);
        eprintln!(
            "{:>28}: {} frames, {} at {} Hz, {} subband predictors estimated; RMS/peak dBFS:{line}",
            s.file,
            frames.len(),
            info.layout,
            info.sample_rate,
            dec.adpcm_estimated()
        );
    }
}

/// XCh: decoding the extension leaves FL, FR, FC and LFE exactly as the core
/// alone decodes them, and changes each surround by exactly −0.7071 × the
/// back centre (the embedded downmix undone, 6.4.1).
#[test]
fn xch_is_consistent_with_its_core() {
    for (_, s) in SAMPLES.iter().filter(|(_, s)| s.decoded.xch) {
        let Some(bytes) = load(s) else { continue };
        let take = |ext| {
            let mut d = decoder(AdpcmFallback::Estimate, ext);
            let r = decode_all(&bytes[..bytes.len().min(4_000_000)], &mut d);
            planar(
                &r.into_iter()
                    .filter_map(Result::ok)
                    .flatten()
                    .collect::<Vec<_>>(),
            )
        };
        let full = take(Extensions {
            xbr: false,
            ..Extensions::ALL
        });
        let core = take(Extensions::NONE);
        // 6.1 = FL FR FC LFE BC SL SR; 5.1(side) = FL FR FC LFE SL SR.
        let n = full[0].len().min(core[0].len());
        for c in 0..4 {
            assert_eq!(
                &full[c][..n],
                &core[c][..n],
                "{}: channel {c} changed by XCh",
                s.file
            );
        }
        let (mut err, mut sig, mut bc) = (0f64, 0f64, 0f64);
        for (fs, cs) in [(5, 4), (6, 5)] {
            for i in 0..n {
                let want = core[cs][i] as f64 - std::f64::consts::FRAC_1_SQRT_2 * full[4][i] as f64;
                err += (full[fs][i] as f64 - want).powi(2);
                sig += (core[cs][i] as f64).powi(2);
                bc += (full[4][i] as f64).powi(2);
            }
        }
        let rel = 10.0 * (err / sig.max(1e-30)).log10();
        eprintln!(
            "{:>28}: surround = core surround − 0.7071·BC to {rel:.0} dB; BC at {:.1} dB of the core surrounds",
            s.file,
            10.0 * (bc / sig.max(1e-30)).log10()
        );
        assert!(rel < -100.0, "{}: surround mismatch {rel:.1} dB", s.file);
    }
}

/// XBR adds residuals: a correction well below the signal it corrects.
#[test]
fn xbr_is_a_small_correction() {
    for (_, s) in SAMPLES.iter().filter(|(_, s)| s.decoded.xbr) {
        let Some(bytes) = load(s) else { continue };
        let take = |xbr| {
            let mut d = decoder(
                AdpcmFallback::Estimate,
                Extensions {
                    xbr,
                    ..Extensions::ALL
                },
            );
            let r = decode_all(&bytes[..bytes.len().min(4_000_000)], &mut d);
            planar(
                &r.into_iter()
                    .filter_map(Result::ok)
                    .flatten()
                    .collect::<Vec<_>>(),
            )
        };
        let (with, without) = (take(true), take(false));
        let (mut d, mut sig) = (0f64, 0f64);
        for (a, b) in with.iter().zip(&without) {
            for (x, y) in a.iter().zip(b) {
                d += (*x as f64 - *y as f64).powi(2);
                sig += (*y as f64).powi(2);
            }
        }
        let rel = 10.0 * (d / sig.max(1e-30)).log10();
        eprintln!(
            "{:>28}: XBR correction at {rel:.1} dB of the signal",
            s.file
        );
        assert!(
            rel < -10.0 && rel > -120.0,
            "{}: XBR correction {rel:.1} dB",
            s.file
        );
    }
}

/// X96: the 96 kHz output is the core's 48 kHz output, interpolated, plus
/// the extension's residuals: compared with the core output band-limited-
/// interpolated to 96 kHz (at the half-sample offset of the two synthesis
/// banks), the difference is small.
#[test]
fn x96_keeps_the_core_band() {
    use std::f64::consts::PI;
    for (_, s) in SAMPLES.iter().filter(|(_, s)| s.decoded.x96) {
        let Some(bytes) = load(s) else { continue };
        let take = |x96| {
            let mut d = decoder(
                AdpcmFallback::Estimate,
                Extensions {
                    x96,
                    ..Extensions::ALL
                },
            );
            let r = decode_all(&bytes[..bytes.len().min(600_000)], &mut d);
            planar(
                &r.into_iter()
                    .filter_map(Result::ok)
                    .flatten()
                    .collect::<Vec<_>>(),
            )
        };
        let (hi, lo) = (take(true), take(false));
        for c in 0..hi.len() {
            let (y96, y48) = (&hi[c], &lo[c]);
            // Hann-windowed sinc, ±64 taps, at core time (j − ½)/2.
            let interp = |t: f64| {
                let m0 = t.floor() as isize;
                (m0 - 63..=m0 + 64)
                    .filter(|m| *m >= 0 && (*m as usize) < y48.len())
                    .map(|m| {
                        let d = t - m as f64;
                        let sinc = if d.abs() < 1e-12 {
                            1.0
                        } else {
                            (PI * d).sin() / (PI * d)
                        };
                        y48[m as usize] as f64 * sinc * (0.5 + 0.5 * (PI * d / 65.0).cos())
                    })
                    .sum::<f64>()
            };
            let (mut err, mut sig) = (0f64, 0f64);
            let end = y96.len().min(2 * y48.len()) - 4096;
            for (j, v) in y96.iter().enumerate().take(end).skip(4096) {
                let want = interp((j as f64 - 0.5) / 2.0);
                err += (*v as f64 - want).powi(2);
                sig += want * want;
            }
            let snr = 10.0 * (sig / err.max(1e-30)).log10();
            eprintln!(
                "{:>28}: ch{c}: 96 kHz output vs interpolated core output: {snr:.1} dB apart",
                s.file
            );
            assert!(
                snr > 30.0,
                "{}: ch{c}: X96 output departs from the core band ({snr:.1} dB)",
                s.file
            );
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}

/// SHA-256 (FIPS 180-4), for checking the downloaded streams.
fn sha256(data: &[u8]) -> [u8; 32] {
    #[rustfmt::skip]
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[test]
fn sha256_known_answers() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(
        hex(&sha256(long)),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// The archive URLs, for the fetch script and anyone checking provenance.
#[test]
fn manifest_lists_every_file_once() {
    let mut files: Vec<&str> = SAMPLES.iter().map(|(_, s)| s.file).collect();
    files.sort();
    files.dedup();
    assert_eq!(files.len(), SAMPLES.len());
    for (path, s) in SAMPLES {
        assert!(path.ends_with(s.file), "{path}");
        assert_eq!(s.sha256.len(), 64);
    }
}
