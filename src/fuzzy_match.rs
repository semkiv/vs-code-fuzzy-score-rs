use crate::score::Score;

use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result};

/// Represents a fuzzy match result.
///
#[derive(Clone, Debug)]
pub struct FuzzyMatch {
    /// Score is a metric of how good a match is: the higher the score the better the match.
    ///
    pub score: Score,

    /// Positions of the matching characters in the target
    ///
    pub positions: Vec<usize>,
}

impl PartialEq for FuzzyMatch {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl PartialOrd for FuzzyMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for FuzzyMatch {}

impl Ord for FuzzyMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl Display for FuzzyMatch {
    #[expect(
        clippy::min_ident_chars,
        reason = "Corresponds to the name used in the trait"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut positions = String::new();
        let mut pos_itr = self.positions.iter().peekable();
        while let Some(pos) = pos_itr.next() {
            positions.push_str(&pos.to_string());
            if pos_itr.peek().is_some() {
                positions.push_str(", ");
            }
        }
        write!(
            f,
            "FuzzyMatch {{ score: {}, positions: [{}] }}",
            self.score, positions
        )
    }
}

// TODO: Docs, Tests
