use crate::errors::arithmetic_overflow_error::ArithmeticOverflowError;

use log::trace;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Mul};

/// Score is used to quantify how good a match is: the higher score the better match.
///
/// # Fields:
///   * a [`NumericRepresentation`] with the actual number
///
// TODO: better name?
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct Score(pub NumericRepresentation);

/// Underlying numeric representation of [`Score`]
///
pub type NumericRepresentation = u32;

impl Score {
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Adds a [`Score`] to the current one while printing a trace message.
    /// The message is sent through the [`trace`] macro,
    /// so whether it is actually printed or not depends on the logging options.
    ///
    /// # Errors
    ///   * [`ArithmeticOverflowError`] if an arithmetic overflow occurs.
    ///
    pub fn traced_add_assign(
        &mut self,
        msg: &str,
        other: Self,
    ) -> Result<(), ArithmeticOverflowError> {
        *self = Add::add(*self, other)?;
        trace!("{msg}, score +{other} (now {self})");
        Ok(())
    }
}

impl Add for Score {
    type Output = Result<Self, ArithmeticOverflowError>;

    fn add(self, rhs: Self) -> Self::Output {
        self.0
            .checked_add(rhs.0)
            .ok_or_else(|| ArithmeticOverflowError::add_overflow(self, rhs))
            .map(Score)
    }
}

impl Mul<u32> for Score {
    type Output = Result<Self, ArithmeticOverflowError>;

    fn mul(self, rhs: u32) -> Self::Output {
        self.0
            .checked_mul(rhs)
            .ok_or_else(|| ArithmeticOverflowError::mul_overflow(self, rhs))
            .map(Score)
    }
}

impl Display for Score {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.0.fmt(f)
    }
}

// TODO: Docs, Tests
