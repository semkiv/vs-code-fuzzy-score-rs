pub mod arithmetic_overflow_error;
pub mod match_bonus_error;

use arithmetic_overflow_error::ArithmeticOverflowError;
use match_bonus_error::MatchBonusError;

use std::error::Error;
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Debug)]
pub enum ScoringError {
    ArithmeticOverflow(ArithmeticOverflowError),
    MatchBonusError(MatchBonusError),
}

impl Display for ScoringError {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::ArithmeticOverflow(err) => {
                write!(f, "Arithmetic overflow: ")?;
                Display::fmt(&err, f)
            }
            Self::MatchBonusError(err) => {
                write!(f, "Error calculating match bonus: ",)?;
                Display::fmt(&err, f)
            }
        }
    }
}

impl Error for ScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ArithmeticOverflow(err) => Some(err),
            Self::MatchBonusError(err) => Some(&err.source_error),
        }
    }
}

impl From<ArithmeticOverflowError> for ScoringError {
    fn from(value: ArithmeticOverflowError) -> Self {
        Self::ArithmeticOverflow(value)
    }
}

impl From<MatchBonusError> for ScoringError {
    fn from(value: MatchBonusError) -> Self {
        Self::MatchBonusError(value)
    }
}

// TODO: Docs, Tests
