mod operands;

use operands::BinaryOpOperands;

use std::error::Error;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Add, Mul};

use crate::errors::arithmetic_overflow_error::operands::Operand;
use crate::errors::arithmetic_overflow_error::operands::add_operands::AddOperands;
use crate::errors::arithmetic_overflow_error::operands::mul_operands::MulOperands;

#[derive(Debug)]
pub enum ArithmeticOverflowError {
    Add(Box<dyn BinaryOpOperands>),
    Mul(Box<dyn BinaryOpOperands>),
}

impl ArithmeticOverflowError {
    pub fn add_overflow<T: Add + Operand + 'static>(lhs: T, rhs: T) -> Self {
        Self::Add(Box::new(AddOperands(lhs, rhs)))
    }

    pub fn mul_overflow<T, F>(lhs: T, rhs: F) -> Self
    where
        T: Mul<F> + Operand + 'static,
        F: Operand + 'static,
    {
        Self::Mul(Box::new(MulOperands(lhs, rhs)))
    }
}

impl Display for ArithmeticOverflowError {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Add(operands) => {
                write!(
                    f,
                    "Adding {} to {} would overflow",
                    operands.lhs(),
                    operands.rhs()
                )
            }
            Self::Mul(operands) => {
                write!(
                    f,
                    "Multiplying {} by {} would overflow",
                    operands.lhs(),
                    operands.rhs()
                )
            }
        }
    }
}

impl Error for ArithmeticOverflowError {}

// TODO: Docs, Tests
