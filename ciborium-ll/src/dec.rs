// SPDX-License-Identifier: Apache-2.0

use super::*;

use ciborium_io::Read;

#[cfg(feature = "half")]
use half::f16;

/// An error that occurred while decoding
#[derive(Clone, Debug)]
pub enum Error<T> {
    /// An error occurred while reading bytes
    ///
    /// Contains the underlying error returned while reading.
    Io(T),

    /// An error occurred while parsing bytes
    ///
    /// Contains the offset into the stream where the syntax error occurred.
    Syntax(usize),
}

impl<T> From<T> for Error<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::Io(value)
    }
}

/// A decoder for deserializing CBOR items
///
/// This decoder manages the low-level decoding of CBOR items into `Header`
/// objects. It also contains utility functions for parsing segmented bytes
/// and text inputs.
pub struct Decoder<R> {
    reader: R,
    offset: usize,
    /// A pushed-back header and the length its encoding would take on the
    /// wire (prefix plus affix), so that `pull()` can hand it back without
    /// re-decoding it.
    buffer: Option<(Header, usize)>,
}

impl<R: Read> From<R> for Decoder<R> {
    #[inline]
    fn from(value: R) -> Self {
        Self {
            reader: value,
            offset: 0,
            buffer: None,
        }
    }
}

impl<R: Read> Read for Decoder<R> {
    type Error = R::Error;

    #[inline]
    fn read_exact(&mut self, data: &mut [u8]) -> Result<(), Self::Error> {
        assert!(self.buffer.is_none());
        self.reader.read_exact(data)?;
        self.offset += data.len();
        Ok(())
    }
}

/// The length of a header's canonical encoding (prefix plus affix).
///
/// This is the length `Encoder::push()` would write for the header, and the
/// length `Title::from(header)` describes, computed without building the
/// `Title`.
#[inline]
fn header_len(header: Header) -> usize {
    #[inline]
    fn int(i: u64) -> usize {
        match i {
            x if x <= 23 => 0,
            x if x <= u8::MAX as u64 => 1,
            x if x <= u16::MAX as u64 => 2,
            x if x <= u32::MAX as u64 => 4,
            _ => 8,
        }
    }

    let len = |l: Option<usize>| l.map_or(0, |x| int(x as u64));

    1 + match header {
        Header::Positive(x) | Header::Negative(x) | Header::Tag(x) => int(x),
        Header::Bytes(x) | Header::Text(x) | Header::Array(x) | Header::Map(x) => len(x),
        Header::Break => 0,
        Header::Simple(x) if x <= 23 => 0,
        Header::Simple(..) => 1,
        // Floats pick the narrowest lossless width; reuse the encoder's rule.
        Header::Float(..) => Title::from(header).1.as_ref().len(),
    }
}

impl<R: Read> Decoder<R> {
    /// Reads one header from the wire.
    ///
    /// Each affix width is read with a `read_exact` of constant length, so
    /// an in-memory reader compiles each arm to one bounds check and one
    /// fixed-size load, and the `Header` is built directly from the prefix
    /// and affix bytes.
    #[inline]
    fn pull_wire(&mut self) -> Result<Header, Error<R::Error>> {
        let offset = self.offset;

        let mut prefix = [0u8; 1];
        self.read_exact(&mut prefix[..])?;
        let major = prefix[0] >> 5;
        let minor = prefix[0] & 0b00011111;

        // The argument of the header: an immediate value, or the following
        // 1, 2, 4 or 8 bytes. Major type 7 reads its affix as a simple value
        // or a float instead.
        let arg: u64 = match minor {
            x if x < 24 => x.into(),
            24 => {
                let mut b = [0u8; 1];
                self.read_exact(&mut b[..])?;
                if major == 7 {
                    return Ok(Header::Simple(b[0]));
                }
                b[0].into()
            }
            25 => {
                let mut b = [0u8; 2];
                self.read_exact(&mut b[..])?;
                if major == 7 {
                    return Ok(Header::Float(f16::from_be_bytes(b).into()));
                }
                u16::from_be_bytes(b).into()
            }
            26 => {
                let mut b = [0u8; 4];
                self.read_exact(&mut b[..])?;
                if major == 7 {
                    return Ok(Header::Float(f32::from_be_bytes(b).into()));
                }
                u32::from_be_bytes(b).into()
            }
            27 => {
                let mut b = [0u8; 8];
                self.read_exact(&mut b[..])?;
                if major == 7 {
                    return Ok(Header::Float(f64::from_be_bytes(b)));
                }
                u64::from_be_bytes(b)
            }
            31 => {
                return match major {
                    2 => Ok(Header::Bytes(None)),
                    3 => Ok(Header::Text(None)),
                    4 => Ok(Header::Array(None)),
                    5 => Ok(Header::Map(None)),
                    7 => Ok(Header::Break),
                    _ => Err(Error::Syntax(offset)),
                };
            }
            _ => return Err(Error::Syntax(offset)),
        };

        let len = |x: u64| usize::try_from(x).map_err(|_| Error::Syntax(offset));

        Ok(match major {
            0 => Header::Positive(arg),
            1 => Header::Negative(arg),
            2 => Header::Bytes(Some(len(arg)?)),
            3 => Header::Text(Some(len(arg)?)),
            4 => Header::Array(Some(len(arg)?)),
            5 => Header::Map(Some(len(arg)?)),
            6 => Header::Tag(arg),
            _ => Header::Simple(arg as u8),
        })
    }

    /// Pulls the next header from the input
    #[inline]
    pub fn pull(&mut self) -> Result<Header, Error<R::Error>> {
        if let Some((header, len)) = self.buffer.take() {
            self.offset += len;
            return Ok(header);
        }

        self.pull_wire()
    }

    /// Push a single header into the input buffer
    ///
    /// # Panics
    ///
    /// This function panics if called while there is already a header in the
    /// input buffer. You should take care to call this function only after
    /// pulling a header to ensure there is nothing in the input buffer.
    #[inline]
    pub fn push(&mut self, item: Header) {
        assert!(self.buffer.is_none());
        let len = header_len(item);
        self.buffer = Some((item, len));
        self.offset -= len;
    }

    /// Gets the current byte offset into the stream
    ///
    /// The offset starts at zero when the decoder is created. Therefore, if
    /// bytes were already read from the reader before the decoder was created,
    /// you must account for this.
    #[inline]
    pub fn offset(&mut self) -> usize {
        self.offset
    }

    /// Process an incoming bytes item
    ///
    /// In CBOR, bytes can be segmented. The logic for this can be a bit tricky,
    /// so we encapsulate that logic using this function. This function **MUST**
    /// be called immediately after first pulling a `Header::Bytes(len)` from
    /// the wire and `len` must be provided to this function from that value.
    ///
    /// The `buf` parameter provides a buffer used when reading in the segmented
    /// bytes. A large buffer will result in fewer calls to read incoming bytes
    /// at the cost of memory usage. You should consider this trade off when
    /// deciding the size of your buffer.
    #[inline]
    pub fn bytes<'a>(&'a mut self, len: Option<usize>) -> Segments<'a, R, crate::seg::Bytes> {
        self.push(Header::Bytes(len));
        Segments::new(self, |header| match header {
            Header::Bytes(len) => Ok(len),
            _ => Err(()),
        })
    }

    /// Process an incoming text item
    ///
    /// In CBOR, text can be segmented. The logic for this can be a bit tricky,
    /// so we encapsulate that logic using this function. This function **MUST**
    /// be called immediately after first pulling a `Header::Text(len)` from
    /// the wire and `len` must be provided to this function from that value.
    ///
    /// The `buf` parameter provides a buffer used when reading in the segmented
    /// text. A large buffer will result in fewer calls to read incoming bytes
    /// at the cost of memory usage. You should consider this trade off when
    /// deciding the size of your buffer.
    #[inline]
    pub fn text<'a>(&'a mut self, len: Option<usize>) -> Segments<'a, R, crate::seg::Text> {
        self.push(Header::Text(len));
        Segments::new(self, |header| match header {
            Header::Text(len) => Ok(len),
            _ => Err(()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `header_len` must agree with the encoder's `Title` for every width
    /// boundary, since `push()` relies on it to restore the stream offset.
    #[test]
    fn header_len_matches_title() {
        let headers = [
            Header::Positive(0),
            Header::Positive(23),
            Header::Positive(24),
            Header::Positive(255),
            Header::Positive(256),
            Header::Positive(65535),
            Header::Positive(65536),
            Header::Positive(u32::MAX as u64),
            Header::Positive(u32::MAX as u64 + 1),
            Header::Positive(u64::MAX),
            Header::Negative(0),
            Header::Negative(u64::MAX),
            Header::Tag(23),
            Header::Tag(24),
            Header::Tag(86),
            Header::Bytes(None),
            Header::Bytes(Some(23)),
            Header::Bytes(Some(24)),
            Header::Text(None),
            Header::Text(Some(300)),
            Header::Array(None),
            Header::Array(Some(70000)),
            Header::Map(None),
            Header::Map(Some(5)),
            Header::Break,
            Header::Simple(0),
            Header::Simple(23),
            Header::Simple(24),
            Header::Simple(255),
            Header::Float(0.0),
            Header::Float(1.5),
            Header::Float(0.1),
            Header::Float(f64::from(f32::MAX)),
            Header::Float(f64::NAN),
            Header::Float(f64::INFINITY),
        ];

        for header in headers {
            let expected = Title::from(header).1.as_ref().len() + 1;
            assert_eq!(header_len(header), expected, "{:?}", header);
        }
    }

    /// Decoding through the push-back buffer must equal decoding from the wire.
    #[test]
    fn push_then_pull_is_identity() {
        let input = b"\x1a\x00\x0f\x42\x40\x63abc\xff";
        let mut decoder = Decoder::from(&input[..]);

        let first = decoder.pull().unwrap();
        assert_eq!(first, Header::Positive(1000000));
        assert_eq!(decoder.offset(), 5);
        decoder.push(first);
        assert_eq!(decoder.offset(), 0);
        assert_eq!(decoder.pull().unwrap(), first);
        assert_eq!(decoder.offset(), 5);

        let text = decoder.pull().unwrap();
        assert_eq!(text, Header::Text(Some(3)));
        decoder.push(text);
        assert_eq!(decoder.offset(), 5);
        assert_eq!(decoder.pull().unwrap(), text);
        assert_eq!(decoder.offset(), 6);
        let mut body = [0u8; 3];
        decoder.read_exact(&mut body[..]).unwrap();
        assert_eq!(&body, b"abc");
        assert_eq!(decoder.pull().unwrap(), Header::Break);
        assert_eq!(decoder.offset(), 10);
    }

    /// Invalid minor codes and indefinite integers report the prefix offset.
    #[test]
    fn syntax_errors_report_prefix_offset() {
        for bad in [0x1cu8, 0x1d, 0x1e, 0x1f, 0x3f, 0xdf] {
            let input = [0x00u8, bad];
            let mut decoder = Decoder::from(&input[..]);
            assert_eq!(decoder.pull().unwrap(), Header::Positive(0));
            match decoder.pull() {
                Err(Error::Syntax(1)) => {}
                other => panic!("{:#04x}: {:?}", bad, other),
            }
        }
    }

    extern crate std;
    use std::format;
    use std::string::{String, ToString};
    use std::vec;
    use std::vec::Vec;

    /// A fixed-size output for the encoder: the crate's `Write` for `Vec<u8>` needs the
    /// `alloc` feature, which the crate's own tests do not enable.
    #[derive(Default)]
    struct Sink {
        bytes: [u8; 16],
        len: usize,
    }

    impl ciborium_io::Write for Sink {
        type Error = ();

        fn write_all(&mut self, data: &[u8]) -> Result<(), ()> {
            let end = self.len + data.len();
            self.bytes
                .get_mut(self.len..end)
                .ok_or(())?
                .copy_from_slice(data);
            self.len = end;
            Ok(())
        }

        fn flush(&mut self) -> Result<(), ()> {
            Ok(())
        }
    }

    /// One header on the wire: the prefix byte and the affix bytes, big endian.
    fn wire(major: u8, minor: u8, affix: &[u8]) -> Vec<u8> {
        let mut bytes = vec![(major << 5) | minor];
        bytes.extend_from_slice(affix);
        bytes
    }

    /// Pulls one header from `bytes` and reports the outcome and the decoder's offset.
    fn pull_one(bytes: &[u8]) -> (Result<Header, String>, usize) {
        let mut decoder = Decoder::from(bytes);
        let result = decoder.pull().map_err(|e| match e {
            Error::Io(_) => "io".to_string(),
            Error::Syntax(offset) => format!("syntax@{offset}"),
        });
        (result, decoder.offset())
    }

    fn encoded(header: Header) -> Vec<u8> {
        let mut encoder = Encoder::from(Sink::default());
        encoder.push(header).unwrap();
        let sink = encoder.into_inner();
        sink.bytes[..sink.len].to_vec()
    }

    /// The header a well-formed (major, minor, argument) decodes to.
    fn expected_header(major: u8, minor: u8, arg: u64) -> Header {
        match major {
            0 => Header::Positive(arg),
            1 => Header::Negative(arg),
            2 => Header::Bytes(Some(arg as usize)),
            3 => Header::Text(Some(arg as usize)),
            4 => Header::Array(Some(arg as usize)),
            5 => Header::Map(Some(arg as usize)),
            6 => Header::Tag(arg),
            _ => match minor {
                0..=24 => Header::Simple(arg as u8),
                25 => Header::Float(f16::from_bits(arg as u16).into()),
                26 => Header::Float(f32::from_bits(arg as u32).into()),
                _ => Header::Float(f64::from_bits(arg)),
            },
        }
    }

    /// Every major type (0-7) with every additional-information value that carries an
    /// argument (0-23 immediate, 24-27 one to eight affix bytes): the decoded header, the
    /// offset after it, the push-back round trip, and, for canonical encodings, that the
    /// encoder writes the same bytes back. Then the same headers back to back in one stream.
    #[test]
    fn every_major_type_and_argument_width_decodes() {
        // (minor, affix, argument) for the four affix widths, each above the previous
        // width's range so the encoding is canonical, plus the all-ones eight-byte argument.
        let widths: [(u8, &[u8], u64); 5] = [
            (24, &[0xf0], 0xf0),
            (25, &[0x3e, 0x00], 0x3e00),
            (26, &[0x3f, 0xc0, 0x00, 0x00], 0x3fc0_0000),
            (27, &[0x3f, 0xf8, 0, 0, 0, 0, 0, 0], 0x3ff8_0000_0000_0000),
            (27, &[0xff; 8], u64::MAX),
        ];
        let mut cases: Vec<(Vec<u8>, Header)> = Vec::new();
        for major in 0u8..8 {
            for minor in 0u8..24 {
                cases.push((
                    wire(major, minor, &[]),
                    expected_header(major, minor, minor as u64),
                ));
            }
            for (minor, affix, arg) in widths {
                cases.push((
                    wire(major, minor, affix),
                    expected_header(major, minor, arg),
                ));
            }
        }
        assert_eq!(cases.len(), 8 * 29);
        let mut stream = Vec::new();
        for (bytes, expected) in &cases {
            let (result, offset) = pull_one(bytes);
            let header = result.unwrap_or_else(|e| panic!("{bytes:02x?}: {e}"));
            if let Header::Float(f) = expected {
                if f.is_nan() {
                    assert!(
                        matches!(header, Header::Float(f) if f.is_nan()),
                        "{bytes:02x?}"
                    );
                    continue;
                }
            }
            assert_eq!(header, *expected, "{bytes:02x?}");
            assert_eq!(offset, bytes.len(), "{bytes:02x?}");
            // The 1.5 floats re-encode at the narrowest width (f16, three bytes); every
            // other case is canonical and must re-encode byte for byte.
            if !matches!(header, Header::Float(..)) {
                assert_eq!(header_len(header), bytes.len(), "{bytes:02x?}");
                assert_eq!(encoded(header), *bytes, "{header:?}");
            } else {
                assert_eq!(header_len(header), 3, "{bytes:02x?}");
                assert_eq!(encoded(header), wire(7, 25, &[0x3e, 0x00]), "{header:?}");
            }
            // Pushed back, the header is handed out again with the offset restored (by the
            // header's canonical length, as `push` documents: 0 for every canonical case).
            let mut decoder = Decoder::from(&bytes[..]);
            let first = decoder.pull().unwrap();
            decoder.push(first);
            assert_eq!(
                decoder.offset(),
                bytes.len() - header_len(header),
                "{bytes:02x?}"
            );
            assert_eq!(decoder.pull().unwrap(), header);
            assert_eq!(decoder.offset(), bytes.len());
            stream.extend_from_slice(bytes);
        }
        let mut decoder = Decoder::from(&stream[..]);
        let mut expected_offset = 0;
        for (bytes, expected) in &cases {
            if matches!(expected, Header::Float(f) if f.is_nan()) {
                continue;
            }
            assert_eq!(decoder.pull().unwrap(), *expected, "{bytes:02x?}");
            expected_offset += bytes.len();
            assert_eq!(decoder.offset(), expected_offset);
        }
        assert!(matches!(decoder.pull(), Err(Error::Io(_))));
    }

    /// Additional information 31: indefinite lengths for byte strings, text strings, arrays
    /// and maps, "break" for major type 7, and a syntax error for integers and tags.
    #[test]
    fn indefinite_lengths_and_break() {
        for (major, expected) in [
            (2u8, Header::Bytes(None)),
            (3, Header::Text(None)),
            (4, Header::Array(None)),
            (5, Header::Map(None)),
            (7, Header::Break),
        ] {
            let bytes = wire(major, 31, &[]);
            assert_eq!(pull_one(&bytes), (Ok(expected), 1), "{bytes:02x?}");
            assert_eq!(encoded(expected), bytes);
            assert_eq!(header_len(expected), 1);
        }
        for major in [0u8, 1, 6] {
            assert_eq!(pull_one(&wire(major, 31, &[])), (Err("syntax@0".into()), 1));
            // The offset in the error is the prefix's position in the stream.
            let mut bytes = wire(0, 24, &[42]);
            bytes.extend(wire(major, 31, &[]));
            let mut decoder = Decoder::from(&bytes[..]);
            assert_eq!(decoder.pull().unwrap(), Header::Positive(42));
            assert!(matches!(decoder.pull(), Err(Error::Syntax(2))));
        }
        // An indefinite text string: chunks with their own headers, closed by break.
        let bytes = b"\x7f\x61a\x62bc\x60\xff";
        let mut decoder = Decoder::from(&bytes[..]);
        assert_eq!(decoder.pull().unwrap(), Header::Text(None));
        assert_eq!(decoder.pull().unwrap(), Header::Text(Some(1)));
        let mut chunk = [0u8; 1];
        decoder.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk, b"a");
        assert_eq!(decoder.pull().unwrap(), Header::Text(Some(2)));
        let mut chunk = [0u8; 2];
        decoder.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk, b"bc");
        assert_eq!(decoder.pull().unwrap(), Header::Text(Some(0)));
        assert_eq!(decoder.pull().unwrap(), Header::Break);
        assert_eq!(decoder.offset(), bytes.len());
        assert!(matches!(decoder.pull(), Err(Error::Io(_))));
        // An indefinite array of indefinite maps, pulled header by header.
        let bytes = b"\x9f\xbf\x00\x01\xff\xbf\xff\xff";
        let mut decoder = Decoder::from(&bytes[..]);
        let headers: Vec<Header> = (0..8).map(|_| decoder.pull().unwrap()).collect();
        assert_eq!(
            headers,
            [
                Header::Array(None),
                Header::Map(None),
                Header::Positive(0),
                Header::Positive(1),
                Header::Break,
                Header::Map(None),
                Header::Break,
                Header::Break,
            ]
        );
    }

    /// Additional information 28, 29 and 30 are reserved: a syntax error at the prefix's
    /// offset for every major type, with only the prefix byte consumed.
    #[test]
    fn reserved_minors_are_syntax_errors_for_every_major() {
        for major in 0u8..8 {
            for minor in [28u8, 29, 30] {
                let bytes = wire(major, minor, &[0xff; 8]);
                assert_eq!(
                    pull_one(&bytes),
                    (Err("syntax@0".into()), 1),
                    "{bytes:02x?}"
                );
                let mut stream = wire(0, 0, &[]);
                stream.extend(wire(3, 2, b"hi"));
                stream.extend(&bytes);
                let mut decoder = Decoder::from(&stream[..]);
                assert_eq!(decoder.pull().unwrap(), Header::Positive(0));
                assert_eq!(decoder.pull().unwrap(), Header::Text(Some(2)));
                let mut text = [0u8; 2];
                decoder.read_exact(&mut text).unwrap();
                assert!(
                    matches!(decoder.pull(), Err(Error::Syntax(4))),
                    "{bytes:02x?}"
                );
                assert_eq!(decoder.offset(), 5);
            }
        }
    }

    /// An affix cut short, at every width and for every major type (and an empty input), is
    /// an I/O error; the prefix byte is the only byte consumed.
    #[test]
    fn truncated_affixes_are_io_errors() {
        assert_eq!(pull_one(&[]), (Err("io".into()), 0));
        for major in 0u8..8 {
            for (minor, width) in [(24u8, 1usize), (25, 2), (26, 4), (27, 8)] {
                for provided in 0..width {
                    let bytes = wire(major, minor, &vec![0x7f; provided]);
                    assert_eq!(pull_one(&bytes), (Err("io".into()), 1), "{bytes:02x?}");
                }
                // The full width decodes.
                let bytes = wire(major, minor, &vec![0x7f; width]);
                let (result, offset) = pull_one(&bytes);
                assert!(result.is_ok(), "{bytes:02x?}: {result:?}");
                assert_eq!(offset, 1 + width);
            }
        }
        // Truncated after a complete header: the first header is intact, the offset stops
        // after the second prefix.
        let bytes = b"\x18\x2a\x1b\x00\x00\x00";
        let mut decoder = Decoder::from(&bytes[..]);
        assert_eq!(decoder.pull().unwrap(), Header::Positive(42));
        assert!(matches!(decoder.pull(), Err(Error::Io(_))));
        assert_eq!(decoder.offset(), 3);
    }

    /// Floats at all three widths, including the values that do not survive narrowing:
    /// infinities, NaN, negative zero, subnormals and the f16 maximum. `canonical` marks the
    /// cases at their narrowest lossless width, where the encoder writes the same bytes back.
    #[test]
    fn float_widths_and_special_values() {
        let cases: [(&[u8], f64, bool); 18] = [
            (&[0xf9, 0x3c, 0x00], 1.0, true),
            (&[0xf9, 0x7b, 0xff], 65504.0, true),
            (&[0xf9, 0x00, 0x01], 5.960_464_477_539_063e-8, true),
            (&[0xf9, 0x80, 0x00], -0.0, true),
            (&[0xf9, 0x7c, 0x00], f64::INFINITY, true),
            (&[0xf9, 0xfc, 0x00], f64::NEG_INFINITY, true),
            (&[0xf9, 0x7e, 0x00], f64::NAN, true),
            (&[0xfa, 0x3f, 0x80, 0x00, 0x00], 1.0, false),
            (&[0xfa, 0x47, 0xc3, 0x50, 0x00], 100000.0, true),
            (
                &[0xfa, 0x00, 0x00, 0x00, 0x01],
                1.401_298_464_324_817e-45,
                true,
            ),
            (&[0xfa, 0x7f, 0x80, 0x00, 0x00], f64::INFINITY, false),
            (&[0xfa, 0x7f, 0xc0, 0x00, 0x00], f64::NAN, false),
            (&[0xfb, 0x3f, 0xf0, 0, 0, 0, 0, 0, 0], 1.0, false),
            (
                &[0xfb, 0x3f, 0xb9, 0x99, 0x99, 0x99, 0x99, 0x99, 0x9a],
                0.1,
                true,
            ),
            (&[0xfb, 0x00, 0, 0, 0, 0, 0, 0, 0x01], 5e-324, true),
            (
                &[0xfb, 0x7f, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                f64::MAX,
                true,
            ),
            (&[0xfb, 0x7f, 0xf0, 0, 0, 0, 0, 0, 0], f64::INFINITY, false),
            (&[0xfb, 0x7f, 0xf8, 0, 0, 0, 0, 0, 0], f64::NAN, false),
        ];
        for (bytes, expected, canonical) in cases {
            let (result, offset) = pull_one(bytes);
            let header = result.unwrap();
            assert_eq!(offset, bytes.len(), "{bytes:02x?}");
            let f = match header {
                Header::Float(f) => f,
                other => panic!("{bytes:02x?}: {other:?}"),
            };
            if expected.is_nan() {
                assert!(f.is_nan(), "{bytes:02x?}");
            } else {
                assert_eq!(f, expected, "{bytes:02x?}");
                assert_eq!(f.is_sign_negative(), expected.is_sign_negative());
            }
            if canonical {
                // Decode and encode are inverse at the narrowest lossless width.
                assert_eq!(encoded(header), bytes, "{bytes:02x?}");
                assert_eq!(header_len(header), bytes.len());
            } else {
                assert!(encoded(header).len() < bytes.len(), "{bytes:02x?}");
                assert_eq!(header_len(header), encoded(header).len());
            }
        }
    }

    /// Simple values: 0-23 in the prefix (false, true, null, undefined among them), 24-255
    /// in one affix byte; an affix byte below 24 is accepted as the value it names.
    #[test]
    fn simple_values_decode_at_both_widths() {
        for value in 0u8..24 {
            let bytes = wire(7, value, &[]);
            assert_eq!(pull_one(&bytes), (Ok(Header::Simple(value)), 1));
            assert_eq!(encoded(Header::Simple(value)), bytes);
        }
        assert_eq!(pull_one(&[0xf4]), (Ok(Header::Simple(simple::FALSE)), 1));
        assert_eq!(pull_one(&[0xf5]), (Ok(Header::Simple(simple::TRUE)), 1));
        assert_eq!(pull_one(&[0xf6]), (Ok(Header::Simple(simple::NULL)), 1));
        assert_eq!(
            pull_one(&[0xf7]),
            (Ok(Header::Simple(simple::UNDEFINED)), 1)
        );
        for value in [24u8, 31, 32, 100, 254, 255] {
            let bytes = wire(7, 24, &[value]);
            assert_eq!(pull_one(&bytes), (Ok(Header::Simple(value)), 2), "{value}");
            assert_eq!(encoded(Header::Simple(value)), bytes);
            assert_eq!(header_len(Header::Simple(value)), 2);
        }
        for value in [0u8, 20, 23] {
            assert_eq!(
                pull_one(&wire(7, 24, &[value])),
                (Ok(Header::Simple(value)), 2)
            );
        }
    }

    /// Eight-byte arguments: the largest integers, tags and lengths, with the wider affixes
    /// holding values that fit a narrower one (accepted, re-encoded canonically).
    #[test]
    fn eight_byte_and_non_canonical_arguments() {
        let max = [0xff; 8];
        assert_eq!(
            pull_one(&wire(0, 27, &max)),
            (Ok(Header::Positive(u64::MAX)), 9)
        );
        assert_eq!(
            pull_one(&wire(1, 27, &max)),
            (Ok(Header::Negative(u64::MAX)), 9)
        );
        assert_eq!(pull_one(&wire(6, 27, &max)), (Ok(Header::Tag(u64::MAX)), 9));
        let big = 1u64 << 32;
        for (major, expected) in [
            (2u8, Header::Bytes(Some(big as usize))),
            (3, Header::Text(Some(big as usize))),
            (4, Header::Array(Some(big as usize))),
            (5, Header::Map(Some(big as usize))),
        ] {
            let bytes = wire(major, 27, &big.to_be_bytes());
            assert_eq!(pull_one(&bytes), (Ok(expected), 9));
            assert_eq!(encoded(expected), bytes);
        }
        // The typed-array tags RedisJSON's CBOR uses take a one-byte affix.
        for tag in [
            tag::TYPED_U8,
            tag::TYPED_I8,
            tag::TYPED_U16_LE,
            tag::TYPED_I16_LE,
        ] {
            let bytes = wire(6, 24, &[tag as u8]);
            assert_eq!(pull_one(&bytes), (Ok(Header::Tag(tag)), 2));
            assert_eq!(encoded(Header::Tag(tag)), bytes);
        }
        // Non-canonical widths decode to the same header the canonical encoding gives.
        for (bytes, expected) in [
            (wire(0, 24, &[5]), Header::Positive(5)),
            (wire(0, 25, &[0, 5]), Header::Positive(5)),
            (wire(0, 26, &[0, 0, 0, 5]), Header::Positive(5)),
            (wire(0, 27, &[0, 0, 0, 0, 0, 0, 0, 5]), Header::Positive(5)),
            (
                wire(4, 27, &[0, 0, 0, 0, 0, 0, 0, 0]),
                Header::Array(Some(0)),
            ),
            (wire(6, 26, &[0, 0, 0, 86]), Header::Tag(86)),
        ] {
            let (result, offset) = pull_one(&bytes);
            assert_eq!(result, Ok(expected), "{bytes:02x?}");
            assert_eq!(offset, bytes.len());
            assert!(encoded(expected).len() < bytes.len());
            // Pushed back, the header restores the canonical length, as `push` documents.
            let mut decoder = Decoder::from(&bytes[..]);
            let header = decoder.pull().unwrap();
            decoder.push(header);
            assert_eq!(decoder.offset(), bytes.len() - header_len(header));
        }
    }
}
