use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};

pub trait Operand: Debug + Display {}

pub trait BinaryOpOperands: Debug {
    fn lhs(&self) -> &(dyn Operand);
    fn rhs(&self) -> &(dyn Operand);
}

#[derive(Debug)]
pub struct AddOperands<T: Add<T> + Operand>(pub T, pub T);

#[derive(Debug)]
pub struct MulOperands<T, F>(pub T, pub F)
where
    T: Mul<F> + Operand,
    F: Operand;

impl<T> Operand for T where T: Debug + Display {}

impl<T: Add<T> + Operand> BinaryOpOperands for AddOperands<T> {
    fn lhs(&self) -> &(dyn Operand) {
        &self.0
    }

    fn rhs(&self) -> &(dyn Operand) {
        &self.1
    }
}

impl<T: Add<T> + Operand + Copy> Copy for AddOperands<T> {}

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

impl<T, F> BinaryOpOperands for MulOperands<T, F>
where
    T: Mul<F> + Operand,
    F: Operand,
{
    fn lhs(&self) -> &(dyn Operand) {
        &self.0
    }

    fn rhs(&self) -> &(dyn Operand) {
        &self.1
    }
}

impl<T, F> Copy for MulOperands<T, F>
where
    T: Mul<F> + Operand + Copy,
    F: Operand + Copy,
{
}

impl<T, F> Clone for MulOperands<T, F>
where
    T: Mul<F> + Operand + Clone,
    F: Operand + Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<T, F> Eq for MulOperands<T, F>
where
    T: Mul<F> + Operand + Eq,
    F: Operand + Eq,
{
}

impl<T, F> PartialEq for MulOperands<T, F>
where
    T: Mul<F> + Operand + PartialEq,
    F: Operand + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T, F> Ord for MulOperands<T, F>
where
    T: Mul<F> + Operand + Ord,
    F: Operand + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.cmp(&other.0) {
            Ordering::Equal => self.1.cmp(&other.1),
            ord @ (Ordering::Less | Ordering::Greater) => ord,
        }
    }
}

impl<T, F> PartialOrd for MulOperands<T, F>
where
    T: Mul<F> + Operand + PartialOrd,
    F: Operand + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.0.partial_cmp(&other.0) {
            Some(Ordering::Equal) => self.1.partial_cmp(&other.1),
            ord => ord,
        }
    }
}

impl<T, F> Hash for MulOperands<T, F>
where
    T: Mul<F> + Operand + Hash,
    F: Operand + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<T, F> Default for MulOperands<T, F>
where
    T: Mul<F> + Operand + Default,
    F: Operand + Default,
{
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}
