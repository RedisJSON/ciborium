// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ciborium::typed_array.

use ciborium::typed_array::{CustomTypedArray, TypedArray};

// ── helpers ───────────────────────────────────────────────────────────────────

fn round_trip<T>(original: &TypedArray<T>) -> TypedArray<T>
where
    T: ciborium::typed_array::TypedArrayElement + PartialEq + core::fmt::Debug,
    TypedArray<T>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let mut buf = Vec::new();
    ciborium::into_writer(original, &mut buf).expect("serialize");
    ciborium::from_reader(buf.as_slice()).expect("deserialize")
}

fn round_trip_custom<T, const TAG: u64>(
    original: &CustomTypedArray<T, TAG>,
) -> CustomTypedArray<T, TAG>
where
    T: bytemuck::Pod + PartialEq + core::fmt::Debug,
    CustomTypedArray<T, TAG>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let mut buf = Vec::new();
    ciborium::into_writer(original, &mut buf).expect("serialize");
    ciborium::from_reader(buf.as_slice()).expect("deserialize")
}

// ── TypedArray round-trips ────────────────────────────────────────────────────

#[test]
fn round_trip_u8() {
    let a = TypedArray(vec![0u8, 1, 127, 255]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_i8() {
    let a = TypedArray(vec![-128i8, -1, 0, 1, 127]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_u16() {
    let a = TypedArray(vec![0u16, 1, 256, 65535]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_i16() {
    let a = TypedArray(vec![-32768i16, -1, 0, 1, 32767]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_u32() {
    let a = TypedArray(vec![0u32, 1, u32::MAX / 2, u32::MAX]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_i32() {
    let a = TypedArray(vec![i32::MIN, -1i32, 0, 1, i32::MAX]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_f32() {
    let a = TypedArray(vec![0.0f32, -1.5, 3.12, f32::MAX, f32::MIN_POSITIVE]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_u64() {
    let a = TypedArray(vec![0u64, 1, u64::MAX / 2, u64::MAX]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_i64() {
    let a = TypedArray(vec![i64::MIN, -1i64, 0, 1, i64::MAX]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_f64() {
    let a = TypedArray(vec![
        0.0f64,
        -1.5,
        3.12,
        f64::MAX,
        f64::MIN_POSITIVE,
    ]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn round_trip_f16() {
    use half::f16;
    let a = TypedArray(vec![
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(-1.5),
        f16::from_f32(3.12),
    ]);
    assert_eq!(round_trip(&a), a);
}

#[test]
fn empty_array_round_trips() {
    let a: TypedArray<u32> = TypedArray(vec![]);
    assert_eq!(round_trip(&a), a);
}

// ── Tag validation ────────────────────────────────────────────────────────────

/// Encode as f32 (tag 85) and try to decode as f64 (tag 86) — must fail.
#[test]
fn tag_mismatch_is_rejected() {
    let original = TypedArray(vec![1.0f32, 2.0]);
    let mut buf = Vec::new();
    ciborium::into_writer(&original, &mut buf).expect("serialize");

    let result: Result<TypedArray<f64>, _> = ciborium::from_reader(buf.as_slice());
    assert!(
        result.is_err(),
        "decoding f32-tagged bytes as f64 must fail"
    );
}

/// Encode as i32 (tag 78) and try to decode as u32 (tag 70) — must fail.
#[test]
fn tag_mismatch_i32_as_u32_rejected() {
    let original = TypedArray(vec![1i32, -1, 0]);
    let mut buf = Vec::new();
    ciborium::into_writer(&original, &mut buf).expect("serialize");

    let result: Result<TypedArray<u32>, _> = ciborium::from_reader(buf.as_slice());
    assert!(result.is_err());
}

// ── Misaligned bytes ──────────────────────────────────────────────────────────

/// Hand-craft a valid CBOR Tag(85, Bytes[13 bytes]) and verify that decoding
/// as TypedArray<f32> fails (13 is not a multiple of 4).
#[test]
fn misaligned_f32_rejected() {
    // CBOR: 0xD8 0x55 — tag 85 (one-byte extra); 0x4D — bstr len 13; 13 zero bytes
    let mut cbor = vec![0xD8u8, 0x55, 0x4D];
    cbor.extend_from_slice(&[0u8; 13]);

    let result: Result<TypedArray<f32>, _> = ciborium::from_reader(cbor.as_slice());
    assert!(
        result.is_err(),
        "13 bytes is not a multiple of 4 — must be rejected"
    );
}

/// Tag(84, Bytes[5 bytes]) — 5 is not a multiple of 2 for f16.
#[test]
fn misaligned_f16_rejected() {
    // 0xD8 0x54 — tag 84; 0x45 — bstr len 5; 5 zero bytes
    let mut cbor = vec![0xD8u8, 0x54, 0x45];
    cbor.extend_from_slice(&[0u8; 5]);

    let result: Result<TypedArray<half::f16>, _> = ciborium::from_reader(cbor.as_slice());
    assert!(
        result.is_err(),
        "5 bytes is not a multiple of 2 — must be rejected"
    );
}

// ── CustomTypedArray ──────────────────────────────────────────────────────────

/// bf16 with a private tag — round-trip via CustomTypedArray.
#[test]
fn custom_tag_bf16_round_trip() {
    use half::bf16;

    const BF16_TAG: u64 = 0x10000;

    let original = CustomTypedArray::<bf16, BF16_TAG>(vec![
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        bf16::from_f32(-3.12),
    ]);
    assert_eq!(round_trip_custom::<bf16, BF16_TAG>(&original), original);
}

/// Verify the tag emitted by CustomTypedArray matches the const generic TAG.
#[test]
fn custom_tag_is_emitted() {
    const MY_TAG: u64 = 99999;

    let original = CustomTypedArray::<u32, MY_TAG>(vec![1u32, 2, 3]);
    let mut buf = Vec::new();
    ciborium::into_writer(&original, &mut buf).expect("serialize");

    let value: ciborium::Value = ciborium::from_reader(buf.as_slice()).expect("decode value");
    match value {
        ciborium::Value::Tag(tag, _) => {
            assert_eq!(tag, MY_TAG, "wrong tag emitted");
        }
        other => panic!("expected Tag, got {other:?}"),
    }
}

/// Mismatched custom tag must be rejected.
#[test]
fn custom_tag_mismatch_rejected() {
    const TAG_A: u64 = 0x10001;
    const TAG_B: u64 = 0x10002;

    let original = CustomTypedArray::<u32, TAG_A>(vec![1u32, 2, 3]);
    let mut buf = Vec::new();
    ciborium::into_writer(&original, &mut buf).expect("serialize");

    let result: Result<CustomTypedArray<u32, TAG_B>, _> = ciborium::from_reader(buf.as_slice());
    assert!(result.is_err(), "wrong custom tag must be rejected");
}

// ── User-defined Pod type ─────────────────────────────────────────────────────

/// Verify that a user-defined element type (implementing bytemuck::Pod) round-trips correctly.
#[test]
fn user_defined_pod_type() {
    /// A simple packed 4-byte RGBA value.
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct Rgba {
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    }

    const RGBA_TAG: u64 = 0x20000;

    let original = CustomTypedArray::<Rgba, RGBA_TAG>(vec![
        Rgba {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        },
        Rgba {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        },
        Rgba {
            r: 0,
            g: 0,
            b: 255,
            a: 128,
        },
    ]);
    assert_eq!(round_trip_custom(&original), original);
}
