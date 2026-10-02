# rivet-dts

[![CI](https://github.com/rivet-transcoder/rivet-dts/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-dts/actions/workflows/ci.yml)

A **DTS Coherent Acoustics core** decoder in Rust: no C, no system
libraries, no build script, nothing to install on a build host. Written
from ETSI TS 102 114 (V1.6.1, cross-checked against V1.2.1), not translated
from any other implementation. On every stream it was checked on (made by
ffmpeg's DTS encoder) it agrees with ffmpeg's decoder to a few parts per
million, relative RMS; the figures are [below](#how-it-is-checked).

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, where it is the DTS decoder: what lets a DTS track from
Matroska (`A_DTS`) or MP4 (`dtsc`, `dtsh`, `dtsl`) be downmixed, filtered or
transcoded to Opus, AAC, MP3, FLAC or ALAC. Usable on its own by anything
that has DTS core frames and wants PCM back.

Published as `rivet-dts`; **imported as `dts`** (`use dts::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
dts = { package = "rivet-dts", git = "https://github.com/rivet-transcoder/rivet-dts", branch = "develop" }
```

## What it decodes

The **core substream** only: up to five full-range channels plus LFE, at up
to 48 kHz, which is what every DTS stream carries and what legacy DTS
decoders play.

| | decoded | refused with `Error::Unsupported` |
|---|---|---|
| **Framing** | core frames with 16-bit big-endian framing (the only framing inside Matroska and MP4), one or more per packet; a DTS-HD extension substream after the core is skipped | termination frames (`FTYPE` = 0); 14-bit and little-endian framing are not read (`Error::NoSync`) |
| **Rates** | 8, 11.025, 12, 16, 22.05, 24, 32, 44.1 and 48 kHz (every core `SFREQ`) | — |
| **Channels** | `AMODE` 0–5 and 7–9: mono, the two-channel modes (dual mono, L+R, sum/difference, LT+RT), C+L+R, C+L+R+S, L+R+SL+SR, C+L+R+SL+SR; with an LFE on the two-channel modes and the 5-channel one | `AMODE` 6 (L+R+S), an LFE on mono, 3.0, 4.0 or quad, `AMODE` 10–15 (they need the XCh / XXCh extension), user-defined arrangements |
| **Tools** | Huffman, block-coded and linear quantisation indices; 6- and 7-bit scale factors with transient splits; joint intensity coding; front and surround sum/difference; dynamic range control; both QMF prototypes; 64× and 128× LFE interpolation | ADPCM prediction (`PMODE` = 1; see below); an encoder revision above `VERNUM` 7 |
| **Ignored** | high-frequency VQ subbands decode as silence (see below); dialog normalisation and the embedded down-mix coefficients are not applied | — |

**Not decoded at all, on purpose:** the extensions — XCh, XXCh and X96
inside the core frame, and the DTS-HD substream after it (DTS-HD High
Resolution and Master Audio, DTS:X). A DTS-HD track decodes as its lossy
core.

Output is interleaved `f32` at ±1.0 full scale, 32 × (`NBLKS` + 1) samples
per channel per frame (512 for the usual 48 kHz frame), in ffmpeg's native
order for the layout, which [`Layout`](src/lib.rs) names:

| `AMODE` | layout | output |
|---|---|---|
| 0 | `mono` | FC |
| 1–4 | `stereo` | FL FR |
| 1–4 + LFE | `2.1` | FL FR LFE |
| 5 | `3.0` | FL FR FC |
| 7 | `4.0` | FL FR FC BC |
| 8 | `quad(side)` | FL FR SL SR |
| 9 | `5.0(side)` | FL FR FC SL SR |
| 9 + LFE | `5.1(side)` | FL FR FC LFE SL SR |

### The two code books ETSI does not print

Two tables of the core are missing from every ETSI edition of the
specification ("Due to its extensive size, this table is not included
here", D.10), and the DTS patents describe them without printing them. The
only copies are in other decoders' source, which this crate does not take
from. So:

- **ADPCM prediction (D.10.1).** A subband coded with `PMODE` = 1 carries
  prediction residuals, and without the 4096-vector coefficient code book
  the prediction cannot be inverted. Such a frame is **refused by name**
  (`Error::Unsupported`, "ADPCM prediction (PMODE = 1 in N subbands) …")
  rather than decoded wrongly; the decoder stays usable and the next frame
  decodes (without that frame's filter-bank history). This matters: ffmpeg's encoder never predicts, so its
  streams decode completely, but a commercial DTS-HD MA track measured on
  2026-09-13 (60 s of its core, 5 637 frames) predicted at least one subband
  in 91 % of its frames, so **most disc-sourced DTS is refused**. A caller
  that would rather keep such a track can pass it through.
- **High-frequency VQ (D.10.2).** Subbands from `VQSUB` up are carried as
  one 10-bit vector index each. The specification allows ignoring them ("the
  VQs for the highest frequency subbands may be ignored without causing
  audible distortion", §5.4.3), and they decode as silence;
  `Decoder::hf_vq_skipped()` says when that has happened. The same
  commercial track VQ-coded subbands 28–31 (21–24 kHz) in every frame.

## How it is checked

- **Against ffmpeg's decoder, as a black box** (`tests/dts_core.rs`; CI
  installs ffmpeg for it). ffmpeg's `dca` encoder makes test streams from
  known signals at test time, ffmpeg decodes them to float PCM, and this
  decoder's output must agree channel by channel to 1e-3 relative RMS, and
  in length to one frame. Without ffmpeg on PATH these tests say so and
  pass, unless `DTS_REQUIRE_FFMPEG` is set (as in CI's oracle job).
- **Unit tests** of the frame-level contracts: the refusals (ADPCM,
  termination frames, encoder revisions, arrangements with no layout)
  trigger by name on synthetic frames, a short packet is an error and never
  zero-filled, the `AMODE` → speaker mapping, the bit reader, the Huffman
  books.
- **Tables**: `tools/dts_gen_tables.py` (see [Provenance](#provenance-and-licensing))
  refuses to emit `src/tables.rs` unless every Huffman book has the
  expected size, is prefix-free and is a complete code (Kraft sum 1), and
  every FIR has 512 taps; the books and FIRs must agree between the V1.6.1
  and V1.2.1 editions.
- **A real stream, on request**: point `RIVET_DTS_SAMPLE` at a raw core
  stream and `real_sample_if_pointed_at_one` reports what the decoder makes
  of it, frame by frame against ffmpeg.

Measured 2026-10-02 against ffmpeg 8.1 on Windows (relative RMS error is this decoder's
output against ffmpeg's, per channel):

| stream (ffmpeg's encoder) | rate | bit rate | worst relative RMS error | largest \|difference\| |
|---|---|---|---|---|
| 5.1(side): sines, pink and brown noise | 48 kHz | 1536 kb/s | 6.0e-6 (LFE and surrounds; fronts 3.1e-6) | 8.3e-7 |
| stereo: sine and white noise | 48 kHz | 768 kb/s | 1.6e-6 | 9.4e-7 |
| stereo | 44.1 kHz | 256 kb/s (coarse quantisers, block codes) | 1.6e-6 | 6.4e-7 |
| stereo | 32 kHz | 192 kb/s | 2.4e-6 | 6.6e-7 |

On the commercial DTS-HD MA core above, the 257 frames that did not predict
and had intact filter history all matched ffmpeg within 1e-3 (its VQ-coded
subbands silent here). Not exercised by any encoder
available to test with: the perfect-reconstruction QMF prototype
(`FILTS` = 1), Huffman-coded `ABITS` and scale-factor differences,
transients (`TMODE`), joint intensity coding and sum/difference coding —
all written from the specification and covered only by unit tests.

## Provenance and licensing

Written from ETSI TS 102 114 V1.6.1 (2019-08), *DTS Coherent Acoustics;
Core and Extensions with Additional Profiles*, which ETSI publishes free of
charge, clause 5 and Annex C (the decoding procedure, followed line by
line). **No DTS implementation's source was read or used** — not FFmpeg's
libavcodec, libdcadec or any other — and ffmpeg was used only as a
command-line tool, to make and decode test streams. Every normative table
(Annex D) was transcribed by `tools/dts_gen_tables.py` from the
`pdftotext -layout` rendering of the V1.6.1 PDF and cross-checked against
the V1.2.1 (2002-12) edition; two printing defects of V1.6.1 (`D65` without
its ±31/±32 rows, `D129` printing level 37 twice) are resolved from V1.2.1,
and the checked-in `src/tables.rs` regenerates byte for byte.

One gain is not in the text: Annex C.3.6 applies an `rScale` to every
synthesis output and never assigns it. The value used, 128·√2, is the one that makes the printed structure unity-gain on the PCM
scale the subband samples are already on; that it agrees with ffmpeg's
output was measured, from the output, not read from its source.

**Patents.** DTS Coherent Acoustics was patented by DTS, Inc. Nothing here
is a licence to any patent, and the authors make no claim about whether
anyone needs one.
"DTS" is a trademark of its owner, used here only to name the format.

## Using it

```rust
// Packets of whole core frames (Matroska A_DTS, MP4 dtsc): one decoder
// per stream, as the filter banks carry history from frame to frame.
let mut dec = dts::Decoder::new();
for packet in packets {
    match dec.decode(packet) {
        Ok(frames) => for f in frames {
            // f.samples: interleaved f32; f.sample_rate; f.channels;
            // f.layout.speakers()
        },
        // Valid DTS this decoder does not decode (ADPCM prediction, …):
        // drop the frame and say why, or pass the track through.
        Err(dts::Error::Unsupported(why)) => eprintln!("{why}"),
        Err(e) => return Err(e),
    }
}

// A raw .dts stream: split it with frame_len.
let mut off = 0;
while off + 8 <= bytes.len() {
    let len = dts::frame_len(&bytes[off..])?;
    let frames = dec.decode(&bytes[off..(off + len).min(bytes.len())])?;
    off += len;
}
```

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
