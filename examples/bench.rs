//! Throughput benchmark: encodes 48 kHz stereo PCM (raw little-endian f32,
//! interleaved) as DTS stereo and 5.1 (the stereo spread over six
//! channels), decodes each result, and prints how many times faster than
//! real time each runs (best of several passes) with a hash of the frames
//! and of the decoded PCM. Further arguments name DTS files to decode.
//!
//! `cargo run --release --example bench -- <pcm.f32> [passes] [filter] [file.dts ...]`

use std::time::Instant;

use dts::{Decoder, Encoder, EncoderConfig, Layout};

fn best<F: FnMut()>(passes: usize, mut f: F) -> f64 {
    (0..passes)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn fnv(hash: &mut u64, bytes: impl IntoIterator<Item = u8>) {
    for b in bytes {
        *hash = (*hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
}

/// Decodes `packets` into `pcm`; returns the seconds of audio.
fn decode_all(packets: &[Vec<u8>], pcm: &mut Vec<f32>) -> f64 {
    let mut dec = Decoder::new();
    dec.set_adpcm_fallback(dts::AdpcmFallback::Estimate);
    let mut secs = 0.0;
    pcm.clear();
    for p in packets {
        // A cut-down sample file's last packet may be cut too.
        for f in dec.decode(p).unwrap_or_default() {
            secs += (f.samples.len() / f.channels) as f64 / f64::from(f.sample_rate);
            pcm.extend_from_slice(&f.samples);
        }
    }
    secs
}

fn pcm_hash(pcm: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut h, pcm.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(args.get(1).expect("pcm file")).unwrap();
    let passes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let only = args.get(3).cloned().unwrap_or_default();
    let stereo: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let secs = stereo.len() as f64 / 2.0 / 48_000.0;
    let six: Vec<f32> = stereo
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[l, r]| {
            [l, r, 0.5 * (l + r), 0.25 * (l + r), 0.5 * (l - r), 0.5 * (r - l)]
        })
        .collect();
    let cases: [(&str, Layout, u32, &[f32]); 2] =
        [("2.0 768k", Layout::Stereo, 768_000, &stereo), ("5.1 1536k", Layout::Surround51Side, 1_536_000, &six)];
    for (name, layout, rate, pcm) in cases {
        if !name.contains(only.as_str()) {
            continue;
        }
        let cfg = EncoderConfig::new(48_000, layout, rate);
        let ch = layout.channels();
        let mut packets = Vec::new();
        let t = best(passes, || {
            let mut enc = Encoder::new(cfg.clone()).unwrap();
            packets.clear();
            for c in pcm.chunks(512 * ch * 4) {
                packets.extend(enc.encode(c).unwrap());
            }
            packets.extend(enc.flush().unwrap());
        });
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        packets.iter().for_each(|p| fnv(&mut h, p.iter().copied()));
        println!("encode  {name:<10} {:7.1} x realtime (stream hash {h:016x})", secs / t);
        let (mut pcm, mut dsecs) = (Vec::new(), 0.0);
        let t = best(passes, || dsecs = decode_all(&packets, &mut pcm));
        println!("decode  {name:<10} {:7.1} x realtime (output hash {:016x})", dsecs / t, pcm_hash(&pcm));
    }
    for path in args.iter().skip(4) {
        let raw = std::fs::read(path).unwrap();
        let data = dts::normalize_framing(&raw).unwrap();
        let mut packets = Vec::new();
        let mut p = 0;
        while p < data.len() {
            match dts::packet_len(&data[p..]) {
                Ok(n) if n > 0 => {
                    let end = (p + n).min(data.len());
                    packets.push(data[p..end].to_vec());
                    p = end;
                }
                _ => break,
            }
        }
        let (mut pcm, mut dsecs) = (Vec::new(), 0.0);
        let t = best(passes, || dsecs = decode_all(&packets, &mut pcm));
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy();
        println!("decode  {name:<30} {:7.1} x realtime (output hash {:016x})", dsecs / t, pcm_hash(&pcm));
    }
}
