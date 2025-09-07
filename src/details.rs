mod match_bonus;
mod scorer_matrix;
mod separator;

use scorer_matrix::ScorerMatrix;

use crate::FuzzyMatch;
use crate::errors::FuzzyScoreError;
use crate::score::Score;

use log::{debug, trace};
use std::fmt::Write as _;

pub fn compute_fuzzy_match(
    query: &str,
    query_length: usize,
    target: &str,
    target_length: usize,
) -> Result<Option<FuzzyMatch>, FuzzyScoreError> {
    // Build a scorer matrix:
    let matrix = ScorerMatrix::new(query, query_length, target, target_length)?;

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
