mod match_bonus;
mod separator;

use separator::Separator;

use crate::FuzzyMatch;
use crate::errors::FuzzyScoreError;
use crate::errors::arithmetic_overflow_error::ArithmeticOverflowError;
use crate::errors::arithmetic_overflow_error::operands::AddOperands;
use crate::score::Score;

use itertools::Itertools as _;
use ndarray::Array2;

use log::{debug, trace};
use std::fmt::Write as _;
use std::ops::Add;

type ScorerMatrix = Array2<ScorerMatrixElement>;

// TODO: make sure that the types implement common traits
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
struct ScorerMatrixElement {
    pub score: Score,
    pub match_sequence_length: usize,
}

pub fn compute_fuzzy_match(
    query: &str,
    query_length: usize,
    target: &str,
    target_length: usize,
) -> Result<Option<FuzzyMatch>, FuzzyScoreError> {
    // Build a scorer matrix:
    let matrix = build_scorer_matrix(query, query_length, target, target_length)?;

    // Restore positions (starting from bottom right of matrix).
    // A match (if any) should be located on one of the diagonals.
    let mut positions = Vec::new();

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "An overflow here means a logic error"
    )]
    let mut query_index = Some(query_length - 1);
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "An overflow here means a logic error"
    )]
    let mut target_index = Some(target_length - 1);

    while let (Some(q_index), Some(t_index)) = (query_index, target_index) {
        #[expect(
            clippy::indexing_slicing,
            reason = "An out-of-bounds index indicates a logic error"
        )]
        let sequence_length = matrix[[q_index, t_index]].match_sequence_length;
        if sequence_length != 0 {
            positions.push(t_index);
            query_index = q_index.checked_sub(1); // go up
        }
        target_index = t_index.checked_sub(1); // go left
    }

    positions.reverse();

    // Print matrices
    trace!("{}", format_scorer_matrix(query, target, &matrix, 4));

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "An overflow here means a logic error"
    )]
    let bottom_right_corner_index = [query_length - 1, target_length - 1];
    #[expect(
        clippy::indexing_slicing,
        reason = "An out-of-bounds index indicates a logic error"
    )]
    let final_score = matrix[bottom_right_corner_index].score;
    debug!(
        "Target: '{target}', query: '{query}', final score: {final_score}, matching positions: {positions:#?}"
    );

    if final_score == Score::zero() {
        return Ok(None);
    }

    Ok(Some(FuzzyMatch {
        score: final_score,
        positions,
    }))
}

fn build_scorer_matrix(
    query: &str,
    query_length: usize,
    target: &str,
    target_length: usize,
) -> Result<ScorerMatrix, FuzzyScoreError> {
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
    let mut matrix = Array2::from_elem(
        [query_length, target_length],
        ScorerMatrixElement::default(),
    );

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
            let diagonal_index =
                left_index.and_then(|[q_idx, t_idx]| q_idx.checked_sub(1).map(|sub| [sub, t_idx]));

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

                    #[expect(
                        clippy::indexing_slicing,
                        reason = "An out-of-bounds index indicates a logic error"
                    )]
                    let current_element = &mut matrix[current_index];
                    current_element.match_sequence_length = usize::from(!score.is_zero());
                    current_element.score = score;
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
                    let score = score_one_pair(query_char, target_char, previous_target_char, 0)?;

                    #[expect(
                        clippy::indexing_slicing,
                        reason = "An out-of-bounds index indicates a logic error"
                    )]
                    let current_element = &mut matrix[current_index];
                    current_element.match_sequence_length = usize::from(!score.is_zero());
                    matrix[current_index].score = if score > Score::zero() {
                        score
                    } else {
                        matrix[left].score
                    };
                }
                (Some(left), Some(diag)) => {
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "An out-of-bounds index indicates a logic error"
                    )]
                    let ScorerMatrixElement {
                        score: diag_score,
                        match_sequence_length,
                    } = matrix[diag];

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

                    #[expect(
                        clippy::indexing_slicing,
                        reason = "An out-of-bounds index indicates a logic error"
                    )]
                    let left_score = matrix[left].score;
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "An out-of-bounds index indicates a logic error"
                    )]
                    let current_element = &mut matrix[current_index];
                    if !score.is_zero() && Add::add(diag_score, score)? >= left_score {
                        current_element.match_sequence_length =
                            match_sequence_length.checked_add(1).ok_or_else(|| {
                                ArithmeticOverflowError::Add(Box::new(AddOperands(
                                    match_sequence_length,
                                    1,
                                )))
                            })?;

                        current_element.score = Add::add(diag_score, score)?;
                    } else {
                        current_element.match_sequence_length = 0;
                        current_element.score = left_score;
                    }
                }
            }
        }
    }

    Ok(matrix)
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

// Formats the matrix as two separate matches and scores matrices:
// `Matches matrix:`
// `    t   a   r   g   e   t`
// `q   M   M   M   M   M   M`
// `u   M   M   M   M   M   M`
// `e   M   M   M   M   M   M`
// `r   M   M   M   M   M   M`
// `y   M   M   M   M   M   M`
// `Score matrix:`
// `    t   a   r   g   e   t`
// `q   S   S   S   S   S   S`
// `u   S   S   S   S   S   S`
// `e   S   S   S   S   S   S`
// `r   S   S   S   S   S   S`
// `y   S   S   S   S   S   S`
fn format_scorer_matrix(query: &str, target: &str, matrix: &ScorerMatrix, indent: usize) -> String {
    let mut matches = String::from("Matches matrix:\n");
    let mut scores = String::from("Scores matrix:\n");

    // print header line, e.g. '    t   a   r   g   e   t'
    matches.push(' ');
    scores.push(' ');

    for chr in target.chars() {
        #[expect(
            clippy::expect_used,
            reason = "See https://doc.rust-lang.org/stable/std/fmt/index.html#formatting-traits:~:text=string%20formatting%20is%20an%20infallible%20operation"
        )]
        write!(matches, "{chr:>indent$}")
            .expect("String formatting should be an infallible operation");
        #[expect(
            clippy::expect_used,
            reason = "See https://doc.rust-lang.org/stable/std/fmt/index.html#formatting-traits:~:text=string%20formatting%20is%20an%20infallible%20operation"
        )]
        write!(scores, "{chr:>indent$}")
            .expect("String formatting should be an infallible operation");
    }
    matches.push('\n');
    scores.push('\n');

    // print the rest
    let mut query_it = query.chars().enumerate().peekable();
    while let Some((query_index, chr)) = query_it.next() {
        matches.push(chr);
        scores.push(chr);
        for (target_index, _) in target.chars().enumerate() {
            let elem = matrix[[query_index, target_index]];
            #[expect(
                clippy::expect_used,
                reason = "See https://doc.rust-lang.org/stable/std/fmt/index.html#formatting-traits:~:text=string%20formatting%20is%20an%20infallible%20operation"
            )]
            write!(
                matches,
                "{:>width$}",
                elem.match_sequence_length,
                width = indent
            )
            .expect("String formatting should be an infallible operation");
            #[expect(
                clippy::expect_used,
                reason = "See https://doc.rust-lang.org/stable/std/fmt/index.html#formatting-traits:~:text=string%20formatting%20is%20an%20infallible%20operation"
            )]
            write!(scores, "{:>width$}", elem.score, width = indent)
                .expect("String formatting should be an infallible operation");
        }

        if query_it.peek().is_some() {
            matches.push('\n');
            scores.push('\n');
        }
    }

    matches.push('\n');
    matches.push_str(&scores);
    matches
}
