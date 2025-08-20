use std::error::Error as StdError;
use std::fmt::{Display, Formatter, Result};
use std::num::TryFromIntError;

#[derive(Clone, Copy, Debug)]
pub struct Error {
    pub sequence_length: usize,
    pub source_error: TryFromIntError,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Matching sequence is too long to be scored (length: {})",
            self.sequence_length
        )
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source_error)
    }
}
