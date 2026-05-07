
pub mod chunked_buffer {
    //! A growable vector which stores data in chunks.
    //!
    //! Space is not reserved for null values.
    //! Primarily useful when data chunks are spread widely through the container.
    //! Uses [`num_bigint::BigUint`] for indexing outside the range of an [`u128`].
    //!
    //! This should be considered infinitely long, limited only by ram space.
    //! So, no index out-of-bounds error may occur from this data structure.
    //! Indexes beyond the last chunk of data simply return the null value assigned at initialization.
    //!
    //! **Note:** [`u128`] indexing fails automatically if the buffer's max address exceeds the [`u128`] max.
    //! As such, it is generally recommended to use [`BigUint`][`num_bigint::BigUint`] wherever possible.

    use std::collections::BTreeMap;
    use std::ops::Index;
    use num_bigint::BigInt;

    /// A growable vector which stores data in chunks.
    pub struct ChunkedBuffer<T> {
        chunks: BTreeMap<BigInt, Box<[T]>>,
        null_value: T,
    }

    impl<T> ChunkedBuffer<T> {

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