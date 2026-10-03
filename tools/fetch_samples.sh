#!/usr/bin/env bash
# Fetch the public DTS sample streams tests/samples.rs checks, into the
# directory given (default: samples/), verifying each against its SHA-256.
# Source: the public sample archive at https://streams.videolan.org/samples/A-codecs/DTS/
# (the files only, used as data). Then:
#   DTS_SAMPLES_DIR=samples cargo test --release --test samples
set -euo pipefail
dir="${1:-samples}"
mkdir -p "$dir"
base="https://streams.videolan.org/samples/A-codecs/DTS"
fetch() { # path sha256
  local path="$1" sum="$2" file
  file="$dir/$(basename "$path")"
  if [ ! -f "$file" ] || ! echo "$sum  $file" | sha256sum -c --status; then
    curl -fsSL --retry 3 -o "$file" "$base/${path// /%20}"
  fi
  echo "$sum  $file" | sha256sum -c
}
fetch "dts/3-1.dts" c21f1dd9a21f96ff198365b22edff20d0c35d8c8c659dd27aac34b720cfb7b9a
fetch "dts/5.1 24bit.dts" 3956047bd9706aa373e3d8b7c8994843374b67351d670ffbb2cd480c3f3190d6
fetch "dts/96-24.dts" 422c8db3496708dbedc67a97b8d8d2652f75e85516d7b46a58ef2df00d48df7b
fetch "dts/ES 6.1 - 5.1 16bit.dts" 57469d05109a39bdc48fc3af77ad873822844d4a90686e58e794f1e2237fc2d4
fetch "dts/ES 6.1 16bit.dts" a04708ac58c70b0da9c0bf63b24c74c4039a398eb2916de715019b1e1f809452
fetch "dts/ES 6.1 24bit.dts" c4017d9426d5e9dae06a3ca681cd11526e438f499a20b5f12a5c4b418d1324f3
fetch "dts/Hi-Res 5.1 24bit.dts" 300cba0e3f2d971921678aa08d19788301d4fa97784b0796e186e554e5998d66
fetch "dts/Hi-Res 6.1 24bit.dts" f182cda9e008073d9f7ccf77257ff8d05665397916c0dbbe39427ca68a7058bb
fetch "dts/Master Audio 2.0 16bit.dts" 34845219924fedc4c633a97c614f464f25011857d11b6ad1919e0c02f1abb3ce
fetch "dts/Master Audio 5.0 96khz.dts" 3702d95a38cba3414968724e7bacb13440b81e79c6be753dc53c2a35813cdb39
fetch "dts/Master Audio 5.1 16bit.dts" 70418af672befaa22b192798f54b864eb74eb5621fdf2a71ca358cffb114e1b0
fetch "dts/Master Audio 7.1 24bit.dts" 0da506ccc59fdef1744bdbe178637199cad05207601e96fbf8d163694fe39c7e
fetch "dts/Master Audio 7.1.dts" 08b6289cceedadfd2e9e7be833e60bb8afd751564cf5e6543fecbc6a3a22f8d9
fetch "dts/dtswavsample14.wav" f6a4889064e9f25eb873d502de8ae388d4a0b5f3776dbb5779d393f2179480e8
fetch "dts/open bitrate.dts" 7069220f675bd6608d37190ddf0d2de1ece570eb057da84131e2bc9a5984d720
fetch "dts/padded.dts" b017346b8f09e3a27a599445d7367879dd2802542bd75b4ce59f0c5e3a5a9c12
fetch "lotr_5.1_768.dts" 6c70137c8d4383668c034bd4993ba7e9cb10044a2165a35fe78bf2316388d8d8
fetch "scissorhands-4.0-48_24.dts" 46c6d087c5e33ca6263c87c2aa752f6167db04f2292075a0831ff9d2264c3150
