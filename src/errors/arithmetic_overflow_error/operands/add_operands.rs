use crate::errors::arithmetic_overflow_error::operands::{BinaryOpOperands, Operand};

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Add;

#[derive(Debug)]
pub struct AddOperands<T: Add<T> + Operand>(pub T, pub T);

impl<T: Add<T> + Operand> BinaryOpOperands for AddOperands<T> {
    fn lhs(&self) -> &(dyn Operand) {
        &self.0
    }

    fn rhs(&self) -> &(dyn Operand) {
        &self.1
    }
}

impl<T: Add<T> + Operand + Copy> Copy for AddOperands<T> {}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "False positive, the trait bound is different from the `Copy` implementation"
)]
impl<T: Add<T> + Operand + Clone> Clone for AddOperands<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<T: Add<T> + Operand + Eq> Eq for AddOperands<T> {}

impl<T: Add<T> + Operand + PartialEq> PartialEq for AddOperands<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T: Add<T> + Operand + Ord> Ord for AddOperands<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.cmp(&other.0) {
            Ordering::Equal => self.1.cmp(&other.1),
            ord @ (Ordering::Less | Ordering::Greater) => ord,
        }
    }
}

impl<T: Add<T> + Operand + PartialOrd> PartialOrd for AddOperands<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.0.partial_cmp(&other.0) {
            Some(Ordering::Equal) => self.1.partial_cmp(&other.1),
            ord => ord,
        }
    }
}

impl<T: Add<T> + Operand + Hash> Hash for AddOperands<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<T: Add<T> + Operand + Default> Default for AddOperands<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

// TODO: Docs, Tests
