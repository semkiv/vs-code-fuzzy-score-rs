pub mod add_operands;
pub mod mul_operands;

use std::fmt::{Debug, Display};

pub trait Operand: Debug + Display {}

pub trait BinaryOpOperands: Debug {
    fn lhs(&self) -> &(dyn Operand);
    fn rhs(&self) -> &(dyn Operand);
}

impl<T> Operand for T where T: Debug + Display {}

// TODO: Docs, Tests
