use crate::error::{AddOverflowError, MulOverflowError};

use log::trace;
use std::fmt::Display;
use std::ops::{Add, Mul};

/// Score is used to quantify how good a match is: the higher score the better match.
///
/// # Fields:
///   * a [`PlainScore`] representing the actual number
///
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Score(pub PlainScore);

/// Underlying numeric representation of [`Score`]
///
pub type PlainScore = u32;

impl Score {
    pub const fn zero() -> Self {
        Score(0)
    }

    pub fn traced_add_assign(&mut self, msg: &str, other: Self) ->  Result<(), AddOverflowError> {
        *self = (*self + other)?;
        trace!("{msg}, score +{other} (now {self})");
        Ok(())
    }
}

impl Add for Score {
    type Output = Result<Self, AddOverflowError>;

    fn add(self, rhs: Self) -> Self::Output {
        self.0
            .checked_add(rhs.0)
            .ok_or_else(|| AddOverflowError { lhs: self, rhs })
            .map(Score)
    }
}

impl Mul<u32> for Score {
    type Output = Result<Self, MulOverflowError>;

    fn mul(self, rhs: u32) -> Self::Output {
        self.0
            .checked_mul(rhs)
            .ok_or_else(|| MulOverflowError {
                score: self,
                factor: rhs,
            })
            .map(Score)
    }
}

impl Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// TODO: Docs, Tests
