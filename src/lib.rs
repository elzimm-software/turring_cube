
pub mod chunked_buffer {
    //! A growable vector which stores data in chunks.
    //!
    //! Space is not reserved for null values.
    //! Primarily useful when data chunks are spread widely through the container.
    //! Uses [`num_bigint::BigInt`] for indexing outside the range of an [`u128`].
    //!
    //! This should be considered infinitely long, limited only by ram space.
    //! So, no index out-of-bounds error may occur from this data structure.
    //! Indexes beyond the last chunk of data simply return the null value assigned at initialization.
    //!
    //! **Note:** [`u128`] indexing fails automatically if the buffer's max address exceeds the [`u128`] max.
    //! As such, it is generally recommended to use [`BigInt`][`num_bigint::BigInt`] wherever possible.

    use std::collections::BTreeMap;
    use std::ops::Index;
    use num_bigint::BigInt;

    struct Chunk<T> {
        data: Box<[T]>,
        start: BigInt,
        len: BigInt,
    }

    /// A growable vector which stores data in chunks.
    pub struct ChunkedBuffer<T> {
        chunks: BTreeMap<BigInt, Chunk<T>>,
        null_value: T,
    }

    impl<T> ChunkedBuffer<T> {
        fn index_in_chunk(&self, idx: &BigInt) -> Option<BigInt> {
            if let Some(c) = self.get_lower_chuck(idx) {
                if idx > &c.1.start && idx < &(&c.1.start + &c.1.len) {
                    return Some(c.0.to_owned());
                }
            }
            None
        }

        fn get_lower_chuck(&self, idx: &BigInt) -> Option<(&BigInt, &Chunk<T>)> {
            self.chunks.range(..idx).next_back()
        }
    }

    /// Dumb indexing.
    ///
    /// Returns null value if no data at index, or index out-of-bounds.
    impl<T, Q: Into<BigInt>> Index<&Q> for ChunkedBuffer<T> {
        type Output = T;

        fn index(&self, index: &Q) -> &Self::Output {
            todo!()
        }
    }

    impl <T> Index<u128> for ChunkedBuffer<T> {
        type Output = Option<T>;

        fn index(&self, index: u128) -> &Self::Output {
            todo!()
        }
    }
}