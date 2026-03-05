// SPDX-License-Identifier: Apache-2.0

//! RFC 8746 typed array support for CBOR, plus custom-tagged byte arrays.
//!
//! # Standard typed arrays — [`TypedArray<T>`]
//!
//! [`TypedArray<T>`] encodes a homogeneous numeric slice as a CBOR tagged byte
//! string per [RFC 8746](https://www.rfc-editor.org/rfc/rfc8746):
//!
//! ```text
//! Tag(rfc8746_tag, Bytes(raw_little_endian_bytes))
//! ```
//!
//! All types use little-endian byte order. `u8` and `i8` are
//! endianness-neutral (single-byte elements). The element type `T` must
//! implement [`TypedArrayElement`], which is sealed to the types listed below.
//!
//! | Type  | RFC 8746 tag |
//! |-------|-------------|
//! | `u8`  | 64          |
//! | `i8`  | 72          |
//! | `u16` | 69          |
//! | `i16` | 77          |
//! | `u32` | 70          |
//! | `i32` | 78          |
//! | `f32` | 85          |
//! | `u64` | 71          |
//! | `i64` | 79          |
//! | `f64` | 86          |
//! | `f16` | 84          |
//!
//! # Custom-tagged byte arrays — [`CustomTypedArray<T, TAG>`]
//!
//! For element types not covered by RFC 8746 (e.g. `bf16`, custom structs
//! packed as raw bytes), use [`CustomTypedArray<T, TAG>`] where `TAG` is a
//! `u64` const generic and `T` implements [`bytemuck::Pod`].
//!
//! ```rust
//! use ciborium::typed_array::CustomTypedArray;
//! type Bf16Array = CustomTypedArray<half::bf16, 0x10000>;
//! ```
//!
//! # Endianness
//!
//! All encode/decode operations assume a **little-endian** target, which is
//! the byte order required by the RFC 8746 LE tags. On little-endian machines
//! (x86, ARM LE, etc.) `bytemuck::cast_slice` produces the correct bytes
//! without any per-element swapping.
//!
//! # Examples
//!
//! ```rust
//! use ciborium::typed_array::TypedArray;
//!
//! let original = TypedArray(vec![1.0f32, 2.5, 3.14]);
//!
//! let mut buf = Vec::new();
//! ciborium::into_writer(&original, &mut buf).unwrap();
//!
//! let decoded: TypedArray<f32> = ciborium::from_reader(buf.as_slice()).unwrap();
//! assert_eq!(decoded.0, vec![1.0f32, 2.5, 3.14]);
//! ```

use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use bytemuck::Pod;
use serde::{de, ser, Deserialize, Serialize};

use crate::tag::Internal;

// ── TypedArrayElement (sealed) ────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Marker for standard RFC 8746 element types with a fixed, well-known tag.
///
/// This trait is sealed: only the types listed in the [module
/// documentation](self) implement it. Use [`CustomTypedArray`] together with
/// [`bytemuck::Pod`] for element types not covered by RFC 8746.
pub trait TypedArrayElement: sealed::Sealed + Pod {
    /// The RFC 8746 CBOR tag number for arrays of this element type.
    const TAG: u64;
}

// ── Macro-generated impls ─────────────────────────────────────────────────────

macro_rules! impl_element {
    ($ty:ty, $tag:expr) => {
        impl sealed::Sealed for $ty {}

        impl TypedArrayElement for $ty {
            const TAG: u64 = $tag;
        }
    };
}

use ciborium_ll::tag as ll_tag;

impl_element!(u8, ll_tag::TYPED_U8);
impl_element!(i8, ll_tag::TYPED_I8);
impl_element!(u16, ll_tag::TYPED_U16_LE);
impl_element!(i16, ll_tag::TYPED_I16_LE);
impl_element!(u32, ll_tag::TYPED_U32_LE);
impl_element!(i32, ll_tag::TYPED_I32_LE);
impl_element!(f32, ll_tag::TYPED_F32_LE);
impl_element!(u64, ll_tag::TYPED_U64_LE);
impl_element!(i64, ll_tag::TYPED_I64_LE);
impl_element!(f64, ll_tag::TYPED_F64_LE);
impl_element!(half::f16, ll_tag::TYPED_F16_LE);

// bf16 has no RFC 8746 standard tag, so it only gets Pod (from the half crate)
// and is usable via CustomTypedArray<half::bf16, TAG>.

// ── RawBytes internal helper ──────────────────────────────────────────────────

/// A `Vec<u8>` that serde-serializes as a CBOR byte string.
struct RawBytes(Vec<u8>);

impl Serialize for RawBytes {
    #[inline]
    fn serialize<S: ser::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for RawBytes {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct BytesVisitor;

        impl<'de> de::Visitor<'de> for BytesVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a byte string")
            }

            #[inline]
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(v.to_vec())
            }

            #[inline]
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(v)
            }
        }

        d.deserialize_bytes(BytesVisitor).map(RawBytes)
    }
}

// ── Shared encode/decode helpers ──────────────────────────────────────────────

/// Zero-copy cast `&[T]` → `Vec<u8>` (native = little-endian on LE targets).
fn pack_bytes<T: Pod>(elems: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(elems).to_vec()
}

/// Decode a raw byte buffer into `Vec<T>`, validating that the byte count is a
/// multiple of `size_of::<T>()` and that the actual CBOR tag matches.
fn unpack_bytes<T: Pod, E: de::Error>(
    bytes: Vec<u8>,
    expected_tag: u64,
    actual_tag: u64,
) -> Result<Vec<T>, E> {
    if actual_tag != expected_tag {
        return Err(de::Error::custom(alloc::format!(
            "expected typed array tag {expected_tag}, got {actual_tag}"
        )));
    }
    let elem_size = size_of::<T>();
    if elem_size == 0 || bytes.len() % elem_size != 0 {
        return Err(de::Error::custom(alloc::format!(
            "byte length {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    // pod_read_unaligned handles arbitrary pointer alignment safely.
    Ok(bytes
        .chunks_exact(elem_size)
        .map(bytemuck::pod_read_unaligned)
        .collect())
}

// ── Public slice serializers ──────────────────────────────────────────────────

/// Serialize a `&[T]` as an RFC 8746 CBOR typed array without cloning to `Vec`.
///
/// Emits `Tag(T::TAG, Bytes(raw_little_endian_elements))`. Useful for
/// implementing CBOR-specific `Serialize` impls for wrapper types that hold a
/// slice or reference to typed data.
pub fn serialize_typed_slice<T, S>(slice: &[T], serializer: S) -> Result<S::Ok, S::Error>
where
    T: TypedArrayElement,
    S: ser::Serializer,
{
    Internal::Tagged(T::TAG, RawBytes(pack_bytes(slice))).serialize(serializer)
}

/// Serialize a `&[T]` as a custom-tagged CBOR byte array without cloning to `Vec`.
///
/// Emits `Tag(TAG, Bytes(raw_little_endian_elements))`. Use this for element
/// types that are not covered by RFC 8746 (e.g. `bf16`).
pub fn serialize_custom_slice<T, const TAG: u64, S>(
    slice: &[T],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    T: Pod,
    S: ser::Serializer,
{
    Internal::Tagged(TAG, RawBytes(pack_bytes(slice))).serialize(serializer)
}

// ── TypedArray ────────────────────────────────────────────────────────────────

/// A typed homogeneous numeric array encoded as an RFC 8746 CBOR tagged byte
/// string.
///
/// Serializing with ciborium produces:
/// ```text
/// Tag(T::TAG, Bytes(raw_little_endian_elements))
/// ```
///
/// Deserializing validates that the CBOR tag matches `T`'s expected RFC 8746
/// tag and that the byte length is a multiple of `size_of::<T>()`.
///
/// For non-standard element types, use [`CustomTypedArray`] instead.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedArray<T: TypedArrayElement>(pub Vec<T>);

impl<T: TypedArrayElement> Serialize for TypedArray<T> {
    fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Internal::Tagged(T::TAG, RawBytes(pack_bytes(&self.0))).serialize(serializer)
    }
}

impl<'de, T: TypedArrayElement> Deserialize<'de> for TypedArray<T> {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Internal::<RawBytes>::deserialize(deserializer)? {
            Internal::Tagged(tag, RawBytes(bytes)) => {
                unpack_bytes::<T, D::Error>(bytes, T::TAG, tag).map(TypedArray)
            }
            Internal::Untagged(_) => Err(de::Error::custom("expected a CBOR tag for typed array")),
        }
    }
}

// ── CustomTypedArray ──────────────────────────────────────────────────────────

/// A typed byte array with a user-supplied CBOR tag.
///
/// Use this for element types that are not covered by RFC 8746, such as
/// `bf16` or application-specific packed structs. `T` must implement
/// [`bytemuck::Pod`] and `TAG` is a `u64` const generic that determines the
/// CBOR tag emitted on serialization and validated on deserialization.
///
/// # Example — bf16 under a private tag
///
/// ```rust
/// use ciborium::typed_array::CustomTypedArray;
///
/// const BF16_TAG: u64 = 0x10000;
///
/// let original = CustomTypedArray::<half::bf16, BF16_TAG>(vec![
///     half::bf16::from_f32(1.0),
///     half::bf16::from_f32(2.0),
/// ]);
/// let mut buf = Vec::new();
/// ciborium::into_writer(&original, &mut buf).unwrap();
///
/// let decoded: CustomTypedArray<half::bf16, BF16_TAG> =
///     ciborium::from_reader(buf.as_slice()).unwrap();
/// assert_eq!(decoded.0, original.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CustomTypedArray<T: Pod, const TAG: u64>(pub Vec<T>);

impl<T: Pod, const TAG: u64> Serialize for CustomTypedArray<T, TAG> {
    fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Internal::Tagged(TAG, RawBytes(pack_bytes(&self.0))).serialize(serializer)
    }
}

impl<'de, T: Pod, const TAG: u64> Deserialize<'de> for CustomTypedArray<T, TAG> {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Internal::<RawBytes>::deserialize(deserializer)? {
            Internal::Tagged(tag, RawBytes(bytes)) => {
                unpack_bytes::<T, D::Error>(bytes, TAG, tag).map(CustomTypedArray)
            }
            Internal::Untagged(_) => Err(de::Error::custom(
                "expected a CBOR tag for custom typed array",
            )),
        }
    }
}
