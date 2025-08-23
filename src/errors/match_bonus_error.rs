use std::error::Error;
use std::fmt::{Display, Formatter, Result};
use std::num::TryFromIntError;

#[derive(Clone, Copy, Debug)]
pub struct MatchBonusError {
    pub sequence_length: usize,
    pub source_error: TryFromIntError,
}

#[expect(
    clippy::min_ident_chars,
    reason = "Corresponds to the name used in the trait"
)]
impl Display for MatchBonusError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Matching sequence is too long to be scored (length: {})",
            self.sequence_length
        )
    }
}

impl Error for MatchBonusError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source_error)
    }
}
