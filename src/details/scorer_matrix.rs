use crate::details::match_bonus;
use crate::details::separator::Separator;
use crate::errors::FuzzyScoreError;
use crate::errors::arithmetic_overflow_error::ArithmeticOverflowError;
use crate::score::Score;

use itertools::Itertools as _;
use ndarray::Array2;

use log::trace;
use std::ops::{Add, Index, IndexMut};

pub struct ScorerMatrix(Array2<ScorerMatrixElement>);

// TODO: make sure that the types implement common traits
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ScorerMatrixElement {
    pub score: Score,
    pub match_sequence_length: usize,
}

impl ScorerMatrix {
    pub fn new(
        query: &str,
        query_length: usize,
        target: &str,
        target_length: usize,
    ) -> Result<Self, FuzzyScoreError> {
        // The matrix is composed of query q and target t.
        // For each index we score q[i] with t[i] and compare that with the previous score.
        // If the score is equal or larger, we keep the match.
        // In addition to the score, we also keep track of the length of the consecutive matches
        // to use as boost for the score.
        //
        //      t   a   r   g   e   t
        //  q   X   X   X   X   X   X
        //  u   X   X   X   X   X   X
        //  e   X   X   X   X   X   X
        //  r   X   X   X   X   X   X
        //  y   X   X   X   X   X   X
        //
        let mut matrix = Self(Array2::from_elem(
            [query_length, target_length],
            ScorerMatrixElement::default(),
        ));

        for (query_index, query_char) in query.chars().enumerate() {
            for (target_index, (previous_target_char, target_char)) in target
                .chars()
                .next()
                .map(|chr| (None, chr))
                .into_iter()
                .chain(
                    target
                        .chars()
                        .tuple_windows()
                        .map(|(prev, curr)| (Some(prev), curr)),
                )
                .enumerate()
            {
                let current_index = [query_index, target_index];
                let left_index = target_index
                    .checked_sub(1)
                    .map(|target_index| [query_index, target_index]);
                let diagonal_index = left_index
                    .and_then(|[q_idx, t_idx]| q_idx.checked_sub(1).map(|sub| [sub, t_idx]));

                match (left_index, diagonal_index) {
                    // This is the left edge of the matrix, there is no previous matching sequence.
                    (None, None) => {
                        let score = if query_index == 0 {
                            // This is the top left element (i.e. there is no match sequence yet).
                            score_one_pair(query_char, target_char, previous_target_char, 0)?
                        } else {
                            // TODO: more comments
                            Score::zero()
                        };

                        let current_element = &mut matrix[current_index];
                        *current_element = compute_left_edge_element(score);
                    }
                    #[expect(
                        clippy::unreachable,
                        reason = "A matrix element cannot have a diagonal neighbor, but no left one"
                    )]
                    // This case is impossible
                    (None, Some(_)) => unreachable!(
                        "An element with a diagonal neighbor, but without a left one is impossible"
                    ),
                    // This is top edge of the matrix (bar the top left element), there is no previous matching sequence.
                    (Some(left), None) => {
                        let score =
                            score_one_pair(query_char, target_char, previous_target_char, 0)?;
                        let left_element = matrix[left];
                        let current_element = &mut matrix[current_index];
                        *current_element = compute_top_edge_element(score, left_element);
                    }
                    (Some(left), Some(diag)) => {
                        let diag_element = matrix[diag];
                        let left_element = matrix[left];
                        let ScorerMatrixElement {
                            score: diag_score,
                            match_sequence_length,
                        } = diag_element;

                        let score = if diag_score.is_zero() {
                            Score::zero()
                        } else {
                            score_one_pair(
                                query_char,
                                target_char,
                                previous_target_char,
                                match_sequence_length,
                            )?
                        };

                        let current_element = &mut matrix[current_index];
                        *current_element = compute_element(score, left_element, diag_element)?;
                    }
                }
            }
        }

        Ok(matrix)
    }
}

impl Index<[usize; 2]> for ScorerMatrix {
    type Output = ScorerMatrixElement;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        self.0.index(index)
    }
}

impl IndexMut<[usize; 2]> for ScorerMatrix {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

fn compute_left_edge_element(score: Score) -> ScorerMatrixElement {
    ScorerMatrixElement {
        score,
        match_sequence_length: usize::from(!score.is_zero()),
    }
}

fn compute_top_edge_element(
    score: Score,
    left_element: ScorerMatrixElement,
) -> ScorerMatrixElement {
    ScorerMatrixElement {
        score: if score > Score::zero() {
            score
        } else {
            left_element.score
        },
        match_sequence_length: usize::from(!score.is_zero()),
    }
}

fn compute_element(
    score: Score,
    left_element: ScorerMatrixElement,
    diagonal_element: ScorerMatrixElement,
) -> Result<ScorerMatrixElement, ArithmeticOverflowError> {
    let left_score = left_element.score;
    let ScorerMatrixElement {
        score: diag_score,
        match_sequence_length,
    } = diagonal_element;

    if !score.is_zero() && Add::add(diag_score, score)? >= left_score {
        return Ok(ScorerMatrixElement {
            score: Add::add(diag_score, score)?,
            match_sequence_length: match_sequence_length
                .checked_add(1)
                .ok_or_else(|| ArithmeticOverflowError::add_overflow(match_sequence_length, 1))?,
        });
    }

    Ok(ScorerMatrixElement {
        score: left_score,
        match_sequence_length: 0,
    })
}

fn score_one_pair(
    query_char: char,
    target_char: char,
    previous_target_char: Option<char>,
    match_sequence_length: usize,
) -> Result<Score, FuzzyScoreError> {
    let query_char_lowercase = query_char.to_lowercase().to_string();
    let target_char_lowercase = target_char.to_lowercase().to_string();

    // No match - no score
    if !considered_equal(&query_char_lowercase, &target_char_lowercase) {
        trace!(
            "'{query_char}' does not match '{target_char}', score {}",
            Score::zero()
        );
        return Ok(Score::zero());
    }

    let mut score = Score::zero();
    // Character match bonus
    score.traced_add_assign(
        &format!("'{query_char}' matches '{target_char}'"),
        match_bonus::base(),
    )?;

    // Consecutive match bonus
    if match_sequence_length > 0 {
        score.traced_add_assign(
            &format!("Consecutive match of length {match_sequence_length}"),
            match_bonus::consecutive(match_sequence_length)?,
        )?;
    }

    // Same case bonus
    if query_char == target_char {
        score.traced_add_assign("Same case", match_bonus::letter_case())?;
    }

    if let Some(previous_target_char) = previous_target_char {
        if let Some(separator) = Separator::from_char(previous_target_char) {
            // After a separator bonus
            score.traced_add_assign(
                "Matches after a separator",
                match_bonus::following_separator(&separator),
            )?;
        } else {
            // Inside word upper case bonus (camel case). We only give this bonus if we're not in a contiguous sequence.
            // For example:
            // NPE => NullPointerException = boost
            // HTTP => HTTP = no boost
            if target_char.is_uppercase() && match_sequence_length == 0 {
                score.traced_add_assign(
                    "Matches camel case inside a word",
                    match_bonus::camel_case(),
                )?;
            }
        }
    } else {
        // Start of word bonus
        score.traced_add_assign("Matches beginning of the word", match_bonus::word_start())?;
    }

    trace!("Final score {score}");
    Ok(score)
}

fn considered_equal(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }

    // TODO: is this a good idea for a general-purpose applications?
    // Special case path separators: ignore platform differences
    if a == "/" || a == "\\" {
        return b == "/" || b == "\\";
    }

    false
}

// TODO: Docs, Tests
