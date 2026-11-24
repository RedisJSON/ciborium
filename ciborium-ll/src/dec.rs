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
}
