use crate::score::Score;

use std::fmt::{Debug, Display};
use std::num::TryFromIntError;

#[derive(Clone, Debug)]
pub enum Error {
    ArithmeticOverflow(ArithmeticOverflowError),
    SequenceTooLong(SequenceTooLongErrorContext),
}

#[derive(Clone, Copy, Debug)]
pub enum ArithmeticOverflowError {
    Add(AddOverflowError),
    Mul(MulOverflowError),
}

#[derive(Clone, Copy, Debug)]
pub struct AddOverflowError {
    pub lhs: Score,
    pub rhs: Score,
}

#[derive(Clone, Copy, Debug)]
pub struct MulOverflowError {
    pub score: Score,
    pub factor: u32,
}

// TODO: move the fields into the error struct directly
#[derive(Clone, Debug)]
pub struct SequenceTooLongErrorContext {
    pub head: String,
    pub tail: String,
    pub length: usize,
    pub source: TryFromIntError,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ArithmeticOverflow(error) => Display::fmt(&error, f),
            Error::SequenceTooLong(context) => {
                write!(
                    f,
                    "Sequence '{}...{}' (length {}) is too long to be scored",
                    context.head, context.tail, context.length
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::ArithmeticOverflow(error) => Some(error),
            Error::SequenceTooLong(context) => Some(&context.source),
        }
    }
}

impl From<ArithmeticOverflowError> for Error {
    fn from(value: ArithmeticOverflowError) -> Self {
        Self::ArithmeticOverflow(value)
    }
}

impl Display for ArithmeticOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Arithmetic overflow occurred when ")?;
        match self {
            ArithmeticOverflowError::Add(context) => {
                write!(f, "adding {} to {}", context.lhs, context.rhs)
            }
            ArithmeticOverflowError::Mul(context) => {
                write!(f, "multiplying {} by {}", context.score, context.factor)
            }
        }
    }
}

impl std::error::Error for ArithmeticOverflowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArithmeticOverflowError::Add(err) => Some(err),
            ArithmeticOverflowError::Mul(err) => Some(err),
        }
    }
}

impl From<AddOverflowError> for ArithmeticOverflowError {
    fn from(value: AddOverflowError) -> Self {
        ArithmeticOverflowError::Add(value)
    }
}

impl From<MulOverflowError> for ArithmeticOverflowError {
    fn from(value: MulOverflowError) -> Self {
        ArithmeticOverflowError::Mul(value)
    }
}

impl Display for AddOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arithmetic overflow occurred when adding {} to {}",
            self.lhs, self.rhs
        )
    }
}

impl std::error::Error for AddOverflowError {}

impl Display for MulOverflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arithmetic overflow occurred when multiplying {} by {}",
            self.score, self.factor
        )
    }
}

impl std::error::Error for MulOverflowError {}

impl SequenceTooLongErrorContext {
    pub fn new(seq: &str, err: TryFromIntError) -> Self {
        const HEAD_LENGTH: usize = 10;
        const TAIL_LENGTH: usize = 10;

        Self {
            head: seq.chars().take(HEAD_LENGTH).collect(),
            tail: seq.chars().rev().take(TAIL_LENGTH).collect(),
            length: seq.len(),
            source: err,
        }
    }
}

// TODO: Docs, Tests
