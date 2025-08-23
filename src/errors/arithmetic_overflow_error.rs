pub mod operands;

use operands::BinaryOpOperands;

use std::fmt::{Debug, Display, Formatter, Result};
use std::error::Error;

#[derive(Debug)]
pub enum ArithmeticOverflowError {
    Add(Box<dyn BinaryOpOperands>),
    Mul(Box<dyn BinaryOpOperands>),
}

impl Display for ArithmeticOverflowError {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Add(operands) => {
                write!(f, "Adding {} to {} would overflow", operands.lhs(), operands.rhs())
            },
            Self::Mul(operands) => {
                write!(f, "Multiplying {} by {} would overflow", operands.lhs(), operands.rhs())
            },
        }
    }
}

impl Error for ArithmeticOverflowError {}
