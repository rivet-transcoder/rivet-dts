//! Run-time choice of instruction set for the vectorisable kernels.
//!
//! The kernels are written so that every output is computed by the same
//! sequence of operations whatever the vector width (each lane is one
//! output; no sum is split across lanes, and no multiply and add are
//! fused), so the compiled copies give bit-identical results and the codec's
//! output does not depend on the CPU. On x86-64 a kernel is compiled twice,
//! for the baseline (SSE2) and for AVX2, and the AVX2 copy runs when the
//! CPU has it; elsewhere (AArch64 has NEON in its baseline) the one copy
//! is used. The `force-scalar` feature keeps only the baseline copy.

/// Defines `fn name(args)` whose body is compiled for the baseline and,
/// on x86-64, for AVX2, chosen at run time.
macro_rules! avx2_or_portable {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn imp($($arg: $ty),*) $(-> $ret)? $body
            #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
            {
                #[target_feature(enable = "avx2")]
                fn avx2($($arg: $ty),*) $(-> $ret)? {
                    imp($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2 (checked just above),
                    // the only requirement of `avx2`.
                    #[allow(unsafe_code)]
                    return unsafe { avx2($($arg),*) };
                }
            }
            imp($($arg),*)
        }
    };
}

pub(crate) use avx2_or_portable;
