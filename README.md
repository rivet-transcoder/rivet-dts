# rivet-dts

[![CI](https://github.com/rivet-transcoder/rivet-dts/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-dts/actions/workflows/ci.yml)

A **DTS Coherent Acoustics** decoder — core, XCh, XXCH, X96, XBR and the
DTS-HD extension substream — and a **DTS core encoder**, in Rust: no C, no
system libraries, no build script. Written from ETSI TS 102 114 (V1.6.1,
tables cross-checked against V1.2.1 and V1.4.1), not translated from any
other implementation, and tested against no other implementation either.

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, where it is the DTS decoder. Usable on its own by anything that
has DTS packets and wants PCM back, or PCM and wants DTS.

Published as `rivet-dts`; **imported as `dts`** (`use dts::…`). One
dependency (`thiserror`), no features.

```toml
[dependencies]
dts = { package = "rivet-dts", git = "https://github.com/rivet-transcoder/rivet-dts", branch = "develop" }
```

## Read this first: the two code books ETSI does not print

DTS Coherent Acoustics uses two vector code books — the **ADPCM prediction
coefficients (D.10.1, 4 096 × 4)** and the **high-frequency subband vectors
(D.10.2, 1 024 × 32)** — that no edition of ETSI TS 102 114 prints ("Due to
its extensive size, this table is not included here"; checked in V1.1.1,
V1.2.1, V1.3.1, V1.4.1, V1.5.1 and V1.6.1, and absent from the DTS
patents checked). The only copies are in other decoders' source, which this crate does
not take from. Everything else in this README follows from that:

- **ADPCM prediction.** Most real DTS predicts some subband in most frames:
  of the 18 public streams below, 51–100 % of frames do. Without D.10.1 a
  predicted frame cannot be decoded exactly. By default it is **refused by
  name** (`Error::Unsupported`, naming D.10.1). The decoder takes the code
  book from a caller who has a lawful copy (`Decoder::set_adpcm_codebook`,
  `AdpcmCodebook::from_be_bytes`) and then decodes such frames exactly.
  Or a caller can opt into `AdpcmFallback::Estimate`, which decodes every
  frame with each transmitted predictor replaced by one estimated from the
  subband's own history, level-limited. **That is concealment, not
  decoding:** a predicted subband is usually tonal, and a tonal residual is
  only reconstructed by the exact predictor. On the crate's own round trip,
  which predicts nearly every subband, it gives ≈ 0 dB SNR; on the public
  streams it decodes every frame at plausible levels (no clipping, channel
  levels as in the exactly decoded frames). Good for keeping a track
  playing; not for transcoding.
- **High-frequency VQ.** Subbands above `VQSUB` decode as silence (§5.4.3
  allows it; `Decoder::hf_vq_skipped` reports it) unless the D.10.2 book is
  supplied (`Decoder::set_hf_vq_codebook`).
- **The encoder** never predicts unless given a code book
  (`EncoderConfig::adpcm_codebook`), so its streams play anywhere.
  `AdpcmCodebook::private_test_book()` is a code book of stable predictors
  built here — **not** D.10.1 — that exercises the prediction paths end to
  end.

## What it decodes

| | decoded | not decoded |
|---|---|---|
| **Framing** | 16-bit big-endian core frames, one or more per packet, each followed by any DTS-HD extension substream frames; extension substream frames on their own when they carry a core (sync 0x02B09261). `normalize_framing` turns raw 16-bit little-endian and 14-bit streams (DTS-CD WAV) into that; `packet_len` splits a raw stream into packets | termination frames (`FTYPE` = 0), refused by name |
| **Core** | every `SFREQ` (8–48 kHz); `AMODE` 0–9 with or without LFE; Huffman, block and linear indices, Huffman-coded `ABITS` and scale factors, transients, joint intensity, sum/difference, dynamic range control, both QMF prototypes, 64×/128× LFE; ADPCM and HF VQ as above | `AMODE` 10–15 (they do not say which channels the core carries), user-defined `AMODE`, `VERNUM` > 7 |
| **XCh** (6.4) | the 6.1 back centre, undoing its −3 dB downmix into the surrounds; joint intensity from core channels | — |
| **XXCH** (6.5) | in the core or the extension substream: channels by loudspeaker mask (up to 32), the core's speakers re-labelled by the core activity mask, embedded downmixes undone (see the open points below) | — |
| **X96** (6.2) | in the core or the extension substream: 64-band synthesis at 88.2/96 kHz, LFE 2× interpolation, noise fill and VQ subbands | — |
| **XBR** (6.3) | in the extension substream: residuals added per channel set | — |
| **DTS-HD** (clause 7) | extension substream header and asset descriptors, CRC-checked; asset 0's XBR/XXCH/X96 components; DTS-HD High Resolution decodes fully, Master Audio decodes its lossy core (plus any XBR/XXCH/X96) | **XLL** (the lossless layer of Master Audio; reported in `CoreInfo::skipped`), **LBR** (DTS Express) |
| **Not applied** | dialog normalisation, embedded down-mix coefficients for down-mixing (the full layout is output) | |

`Decoder::set_extensions` chooses which extensions to decode
(`set_core_only(true)` is a legacy core decoder). Output is interleaved
`f32` at ±1.0 full scale, 32 × (`NBLKS` + 1) samples per channel per frame
(twice that with X96), in canonical speaker order (WAVE's), named
by `Layout`: `mono`, `stereo`, `2.1`, `3.0`, `3.0(back)`, `3.1`, `4.0`,
`4.1`, `quad(side)`, `5.0(side)`, `5.1(side)`, `6.0`, `6.1`, `7.0`, `7.1`,
or `Custom` for any other speaker set. `Speaker` and `Layout` are
`#[non_exhaustive]`.

## What it encodes

`Encoder` writes core frames: every core arrangement (mono, stereo, 3.0,
3.0(back), 4.0, quad(side), 5.0(side), each with or without LFE), 48, 44.1
or 32 kHz, any Table 5-7 bit rate from 32 to 1 536 kb/s (1 411.2 kb/s
included), 512 samples per frame. A 32-band analysis bank matched to the
synthesis bank, scale factors and transients, greedy noise-to-mask bit
allocation with exact bit costs, the cheapest entropy code per quantiser,
the LFE decimated 64×, ADPCM where it pays when given a code book, and rate
control by construction (every frame exactly fills its constant size).
`CPF` = 0, so core frames carry no CRC words (the specification says
"should always be set to 0"); sync words and `DSYNC` are as specified. Not
used: HF VQ, joint intensity, sum/difference, Huffman-coded `ABITS`.

## How it is checked

One other implementation is used, as a black box: libdca's `dcadec`
command-line tool decodes this crate's encoder's streams for comparison
(below). Its source was not read. ffmpeg is not used, as a library, a
binary, a test oracle or a source of test data.

**Round trips through the encoder and decoder** (`tests/encoder_roundtrip.rs`),
1 s of sines, multitone, noise and a chirp per channel, SNR per channel
after the encoder's stated delay (512 samples):

| configuration | SNR per channel |
|---|---|
| 48 kHz, 1 536 kb/s: mono / stereo / 3.0 / 4.0, quad / 5.0, 5.1 | 138 / 125–129 / 80–83 / 66–70 / 58–62 dB |
| 44.1 kHz, 1 411.2 kb/s, same layouts | 138 / 124–126 / 79–83 / 65–68 / 56–61 dB |
| LFE (64× decimation and interpolation) | 44 dB (48 kHz), 41.5 dB (44.1 kHz) |
| 5.1 at 768 kb/s | 29–36 dB |
| stereo 384 kb/s / 256 kb/s at 44.1 / 192 kb/s at 32 kHz; mono 128 kb/s | 53 / 39 / 36; 30 dB |
| stereo tone, 384 kb/s / 256 kb/s, ADPCM with a shared code book | 78 / 64 dB (54 / 40 dB without prediction) |

Frequency response at full rate: within ±0.005 dB from 50 Hz to 20 kHz
(48 and 44.1 kHz); LFE within ±0.15 dB from 20 to 120 Hz. Every frame is
checked for sync, `FSIZE` and the exact constant size.

**Public DTS streams** (`tests/samples.rs`): 18 streams fetched as data
from the public sample archive at <https://streams.videolan.org/samples/A-codecs/DTS/>
by `tools/fetch_samples.sh` and checked against their SHA-256 (listed in
the test): DTS 4.0 and 5.1, DTS-ES 6.1 (XCh), 96/24 (X96), DTS-HD High
Resolution 5.1/6.1 (XBR, XXCH, XCh), DTS-HD Master Audio 2.0–7.1 cores, a
14-bit DTS-CD WAV, an open-rate and a padded stream. No reference output is
published with them, so what is checked is what can be:

- every packet decodes, or is refused for ADPCM and nothing else (51–100 %
  of packets per stream are refused);
- with `AdpcmFallback::Estimate`, every packet of every stream decodes, to
  the layout and rate the stream declares, finite and below full scale;
- **XCh** leaves FL, FR, FC and LFE bit-identical to the core alone and
  moves each surround by exactly 0.7071 × the back centre (to −149 dB);
- **XBR** is a correction 46–57 dB below the signal;
- **X96**'s 96 kHz output is 58–64 dB from the core's 48 kHz output
  band-limited-interpolated (the X96 residuals and the band above 24 kHz
  are the difference).

CI fetches the streams and runs these (`DTS_SAMPLES_DIR`,
`DTS_REQUIRE_SAMPLES`); locally they skip without the files.

**Spec-derived unit tests**: C.3.6 is shown, tap for tap, to be the
cosine-modulated bank `64·s_k·g[n]·cos(π/32·(k+½)·(n+16.5))` for both D.8
prototypes, and random subband input matches the direct convolution; the
X96 bank matches its direct form and synthesises core subbands to the
core's output interpolated (6.2.3); inverse ADPCM matches a float rendering
of C.3.3 across subsubframes and frames (`HFLAG`); every Huffman book
round-trips; block codes decode the C.3.2 example; the VQ path, the D.9
prototype (linear phase, unit gain) and the D.11 grid; CRC-16 (Annex B)
known answers; the encoder's analysis+synthesis reconstruction (148 dB
with `FILTS` = 1, 85 dB with `FILTS` = 0).

**Against libdca's decoder, as a black box** (`tests/dcadec.rs`; CI
installs `libdca-utils` and sets `DTS_REQUIRE_DCADEC`, so a missing tool
fails the job instead of skipping). The encoder's streams — tones, a
multitone, noise and a chirp per channel; 48, 44.1 and 32 kHz; full and
reduced rates; transient detection on and off — are decoded by both, and
per channel the outputs must be sample-aligned, at the same level, and
agree to 1e-4 relative RMS once the level is divided out. Measured on
ubuntu-latest (libdca 0.0.7), 41 streams:

| | gain (theirs / ours) | relative RMS after gain |
|---|---|---|
| full rate (1 536 / 1 411.2 / 1 024 kb/s) | within 1e-5, but mono 0.99864 and 2.1 1.0002 | ≤ 4.3e-5 |
| 128–768 kb/s | within 3e-6 | ≤ 2e-7 |
| the LFE, everywhere | 1 | 0 |

The full-rate level difference where one or two channels get the finest
quantisers is consistent with D.2.1's step sizes being printed as integers
× 2^-22 (21, 42, 84, … for `ABITS` 26, 25, 24, …), i.e. only to a part in a
thousand; this crate uses the printed values. Stereo is compared through
the tool's 16-bit two-channel output (`-o wav`), within its rounding.

The encoder writes `FILTS` = 0 by default: C.3.6 calls that prototype the
lossy one (`raCoeffLossy`, against `raCoeffLossLess` for `FILTS` = 1), it
is what every public stream uses, and libdca decodes `FILTS` = 1 streams to
garbage (probed on this encoder's streams: with the bit set its output is
unrelated to the signal whichever bank the encoder analysed with; with it
clear the two decoders agree as above). `perfect_reconstruction: true`
still writes `FILTS` = 1 streams, which this crate decodes per the
specification. Not covered by the tool: `FILTS` = 1, mono+LFE, and 3–5
channel layouts without the LFE (its WAV output for those mixes channels
or adds one). Those are covered by the round trips only.

## Open points in the specification

Where the text is ambiguous the choice made is documented in the code:

- **X96 synthesis.** ETSI prints the D.9 prototype but no 64-band
  structure; the one used is C.3.6's direct form generalised (gain 128,
  phase `n + 32.5`), the generalisation under which 6.2.3's "core subbands
  in the lower 32 bands give the interpolated core" holds. The 96 kHz
  output is half a 96 kHz sample later than the core's.
- **X96 `SEL96` with `HIGHRESFLAG96K`.** Table 6-4 restarts the
  high-resolution selector loop at the 17-level entry; it is read once.
  Table 6-10's `N = 16·nSSC` is read as 8 samples per subsubframe, as its
  own unpacking loop and the frame duration require.
- **XXCH downmix coefficients** are 7-bit codes (Table 6-23) whose mapping
  is not given; they are read as a sign bit plus a C.6 6-bit table code,
  and the 6-bit scale as a C.6 code, applied as
  `channel = downmix / scale − Σ coefficient × extension channel`.
  Untested: no public stream carries an embedded XXCH downmix.
- **XBR** residuals are added after joint intensity and sum/difference.
- **Huffman-coded `ABITS`**: D.5.6 labels its levels 1–12 as `ABITS`
  values and they are used as such. This crate's encoder does not use
  them; an earlier version of this crate was compared, on streams that
  do, with another decoder to ~1e-6 relative RMS, a comparison no longer
  run. The public streams use them and decode to plausible levels.
- **`ADJ`** is indexed by `ABITS` (Table 5-29 writes
  `arADJ[ch][SEL[ch][nABITS-1]]`, which cannot be meant).
- **XCh's** downmix gain is taken as exactly 1/√2 (D.11's −3 dB entry).
- The asset descriptor's last two flags are read only when its size leaves
  room for them (older streams end before them).

## Provenance and licensing

Written from ETSI TS 102 114 V1.6.1 (2019-08), *DTS Coherent Acoustics;
Core and Extensions with Additional Profiles*, which ETSI publishes free of
charge. **No DTS implementation's source was read or used** — not FFmpeg's
libavcodec, libdcadec or any other. Every normative table (Annex D) was
transcribed by `tools/dts_gen_tables.py` from the `pdftotext -layout`
rendering of the V1.6.1 PDF and cross-checked against V1.2.1 (2002-12) and,
for D.9 and D.11, V1.4.1 (2012-11); printing defects (`D65` without its
±31/±32 rows and `D129` printing 37 twice in V1.6.1; D.9 row 160 printed
twice in every edition; V1.4.1's D.11 inverse column one row late) are
resolved and documented, and the checked-in `src/tables.rs` regenerates
byte for byte. C.3.6's undefined `rScale` is 128·√2, the value that makes
the printed structure unity-gain.

The public sample streams are used only as data and are not distributed
here.

**Patents.** DTS Coherent Acoustics was patented by DTS, Inc. Nothing here
is a licence to any patent, and the authors make no claim about whether
anyone needs one. "DTS" is a trademark of its owner, used here only to name
the format.

## Using it

```rust
// Packets (Matroska A_DTS, MP4 dtsc/dtsh/dtsl): one decoder per stream.
let mut dec = dts::Decoder::new();
// dec.set_adpcm_codebook(Some(book));      // if you hold D.10.1
// dec.set_adpcm_fallback(dts::AdpcmFallback::Estimate); // or conceal
for packet in packets {
    match dec.decode(packet) {
        Ok(frames) => for f in frames {
            // f.samples: interleaved f32; f.sample_rate; f.channels;
            // f.layout.speakers()
        },
        // Valid DTS this decoder does not decode (ADPCM without D.10.1, …):
        // drop the frame and say why, or pass the track through.
        Err(dts::Error::Unsupported(why)) => eprintln!("{why}"),
        Err(e) => return Err(e),
    }
}

// A raw .dts/.dtshd/.wav stream: normalise the framing, split into packets.
let bytes = dts::normalize_framing(&raw)?;
let mut off = 0;
while off + 8 <= bytes.len() {
    let len = dts::packet_len(&bytes[off..])?;
    let frames = dec.decode(&bytes[off..(off + len).min(bytes.len())])?;
    off += len;
}

// Encoding: interleaved f32 in the layout's speaker order.
let mut enc = dts::Encoder::new(dts::EncoderConfig::new(48_000, dts::Layout::Surround51Side, 1_536_000))?;
let mut out: Vec<Vec<u8>> = enc.encode(&pcm)?;
out.extend(enc.flush()?);
```

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
