use crate::match_bonus::error::Error as MatchBonusError;
use crate::score::Score;

use std::error::Error as StdError;
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Clone, Debug)]
pub enum Error {
    ArithmeticOverflow(ArithmeticOverflowError),
    MatchBonusError(MatchBonusError),
}

#[derive(Clone, Copy, Debug)]
pub enum ArithmeticOverflowError {
    Add(Operands<Score, Score>),
    Mul(Operands<Score, u32>),
}

#[derive(Clone, Copy, Debug)]
pub struct Operands<L, R>(pub L, pub R);

impl Display for Error {
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

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::ArithmeticOverflow(err) => Some(err),
            Self::MatchBonusError(err) => Some(&err.source_error),
        }
    }
}

impl From<ArithmeticOverflowError> for Error {
    fn from(value: ArithmeticOverflowError) -> Self {
        Self::ArithmeticOverflow(value)
    }
}

impl From<MatchBonusError> for Error {
    fn from(value: MatchBonusError) -> Self {
        Self::MatchBonusError(value)
    }
}

impl Display for ArithmeticOverflowError {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Add(Operands(lhs, rhs)) => {
                write!(f, "Adding {lhs} to {rhs} would overflow")
            }
            Self::Mul(Operands(score, factor)) => {
                write!(f, "Multiplying {score} by {factor} would overflow")
            }
        }
    }
}

impl StdError for ArithmeticOverflowError {}

// TODO: Docs, Tests
