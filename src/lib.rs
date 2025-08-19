// Based on Visual Studio Code fuzzy matching algorithm
// see https://github.com/microsoft/vscode/blob/648dbbe9a59ab4cf843d9e37f64153b9f0793c15/src/vs/base/common/fuzzyScorer.ts

pub mod error;
pub mod fuzzy_match;
pub mod score;

mod match_bonus;
mod separator;

use error::Error;
use fuzzy_match::FuzzyMatch;
use score::Score;
use separator::Separator;

use itertools::Itertools as _;
use log::{debug, trace};
use ndarray::Array2;

use std::convert::Into;
use std::fmt::{Display, Write as _};

use crate::error::ArithmeticOverflowError;

/// Contains main part of the matching and scoring logic.
///
/// Matches `query` against `target`.
/// If there's a match returns [`Some<FuzzyMatch>`] containing the final score
/// and the positions of the matching characters in `target`, [`None`] otherwise.
/// Also returns [`None`] if either `query` or `target` is empty or `target` is shorter than `query`.
///
/// # Examples:
///
/// ```
/// // there's a match
/// let m = vscode_fuzzy_score_rs::fuzzy_match("baa", "foobarbaz").unwrap();
/// assert_eq!(m.score.0, 11);
/// assert_eq!(*m.positions, vec![3, 4, 7]);
/// ```
///
/// ```
/// // no match
/// let m = vscode_fuzzy_score_rs::fuzzy_match("foo", "barbaz");
/// assert!(m.is_none());
/// ```
///
/// ```
/// // target is too short
/// let m = vscode_fuzzy_score_rs::fuzzy_match("foobar", "bar");
/// assert!(m.is_none());
/// ```
///
#[must_use]
pub fn fuzzy_match(query: &str, target: &str) -> Result<Option<FuzzyMatch>, Error> {
    if query.is_empty() {
        debug!("Query is empty");
        return Ok(None);
    }

    if target.is_empty() {
        debug!("Target is empty");
        return Ok(None);
    }

    let target_length = target.chars().count();
    let query_length = query.chars().count();

    if target_length < query_length {
        debug!(
            "Query '{query}' (length {query_length}) is too long for target '{target}' (length {target_length})"
        );
        return Ok(None); // impossible for query to be contained in target
    }

    compute_fuzzy_match(query, target)
}

// TODO: make this return Result and report arithmetic (and potentially other) errors if any
fn compute_fuzzy_match(query: &str, target: &str) -> Result<Option<FuzzyMatch>, Error> {
    // Build a scorer matrix:
    // The matrix is composed of query q and target t.
    // For each index we score q[i] with t[i] and compare that with the previous score.
    // If the score is equal or larger, we keep the match.
    // In addition to the score, we also keep the length of the consecutive matches to use as boost for the score.
    //
    //      t   a   r   g   e   t
    //  q   X   X   X   X   X   X
    //  u   X   X   X   X   X   X
    //  e   X   X   X   X   X   X
    //  r   X   X   X   X   X   X
    //  y   X   X   X   X   X   X
    //
    let target_length = target.chars().count();
    let query_length = query.chars().count();
    let mut matches = Array2::zeros([query_length, target_length]);
    let mut scores = Array2::from_elem([query_length, target_length], Score::zero());

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
            let left_index = target_index.checked_sub(1).map(|x| [query_index, x]);
            let diagonal_index = query_index
                .checked_sub(1)
                .zip(target_index.checked_sub(1))
                .map(Into::<[usize; 2]>::into);

            let match_sequence_length = diagonal_index
                .and_then(|idx| matches.get(idx))
                .copied()
                .unwrap_or(0);

            // If we are not matching on the first query character any more, we only produce a
            // score if we had a score previously for the last query index (by looking at the diagonal score).
            // This makes sure that the query always matches in sequence on the target.
            // For example given a target of "ede" and a query of "de",
            // we would otherwise produce a wrong high score
            // for query[1] ("e") matching on target[0] ("e") because of the "beginning of word" boost.
            #[expect(
                clippy::indexing_slicing,
                reason = "If diagonal index is not None, it must be valid, otherwise it a logic error"
            )]
            let score = if query_index == 0
                || diagonal_index.is_some_and(|idx| scores[idx] != Score::zero())
            {
                score_one_pair(
                    query_char,
                    target_char,
                    previous_target_char,
                    match_sequence_length,
                )?
            } else {
                Score::zero()
            };

            // We have a score and it's equal or larger than the left score (if one exists).
            // Match: sequence continues growing from previous diag value.
            // Score: increases by diag score value.
            if score > Score::zero()
                && (left_index
                    .zip(diagonal_index)
                    .map_or(Ok(true), |(left, diag)| {
                        (scores[diag] + score).map(|sum| sum >= scores[left])
                    })
                    .map_err(ArithmeticOverflowError::from)?)
            {
                matches[current_index] = match_sequence_length + 1;
                // TODO: simplify?
                scores[current_index] = diagonal_index
                    .map_or(Ok(score), |index| scores[index] + score)
                    .map_err(ArithmeticOverflowError::from)?;
            }
            // We either have no score or the score is lower than the left score.
            // Match: reset to 0.
            // Score: pick up from left hand side.
            else {
                matches[current_index] = 0;
                scores[current_index] = left_index.map_or(Score::zero(), |index| scores[index]);
            }
        }
    }

    // Restore positions (starting from bottom right of matrix)
    let mut positions = Vec::new();
    let mut query_index_it = (0..query_length).rev().peekable();
    let mut target_index_it = (0..target_length).rev().peekable();
    while query_index_it.peek().is_some() && target_index_it.peek().is_some() {
        let query_index = *query_index_it.peek().unwrap();
        let target_index = *target_index_it.peek().unwrap();
        let current_index = [query_index, target_index];
        let current_match = matches[current_index];
        if current_match != 0 {
            positions.push(target_index);
            query_index_it.next(); // go up
        }
        target_index_it.next(); // go left
    }
    positions.reverse();

    // Print matrices
    trace!(
        "{}",
        format_matrix("Matches matrix:", query, target, &matches, 4)
    );
    trace!(
        "{}",
        format_matrix("Scores matrix:", query, target, &scores, 4)
    );

    let final_score = scores[[query_length - 1, target_length - 1]];
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

fn score_one_pair(
    query_char: char,
    target_char: char,
    previous_target_char: Option<char>,
    match_sequence_length: usize,
) -> Result<Score, Error> {
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
    score.traced_add_assign(&format!("'{query_char}' matches '{target_char}'"), match_bonus::base()).map_err(ArithmeticOverflowError::from)?;

    // Consecutive match bonus
    if match_sequence_length > 0 {
        score.traced_add_assign(&format!("Consecutive match of length {match_sequence_length}"), match_bonus::consecutive(match_sequence_length)?).map_err(ArithmeticOverflowError::from)?;
    }

    // Same case bonus
    if query_char == target_char {
        score.traced_add_assign("Same case", match_bonus::letter_case()).map_err(ArithmeticOverflowError::from)?;
    }

    if let Some(previous_target_char) = previous_target_char {
        if let Some(separator) = Separator::from_char(previous_target_char) {
            // After a separator bonus
            score.traced_add_assign(
                "Matches after a separator",
                match_bonus::following_separator(&separator),
            ).map_err(ArithmeticOverflowError::from)?;
        } else {
            // Inside word upper case bonus (camel case). We only give this bonus if we're not in a contiguous sequence.
            // For example:
            // NPE => NullPointerException = boost
            // HTTP => HTTP = no boost
            if target_char.is_uppercase() && match_sequence_length == 0 {
                score.traced_add_assign(
                    "Matches camel case inside a word",
                    match_bonus::camel_case(),
                ).map_err(ArithmeticOverflowError::from)?;
            }
        }
    } else {
        // Start of word bonus
        score.traced_add_assign(
            "Matches beginning of the word",
            match_bonus::word_start()
        ).map_err(ArithmeticOverflowError::from)?;
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

// formats matrix like so:
// `msg:`
// `    t   a   r   g   e   t`
// `q   X   X   X   X   X   X`
// `u   X   X   X   X   X   X`
// `e   X   X   X   X   X   X`
// `r   X   X   X   X   X   X`
// `y   X   X   X   X   X   X`
fn format_matrix<T: Display>(
    msg: &str,
    query: &str,
    target: &str,
    matrix: &Array2<T>,
    indent: usize,
) -> String {
    // print 'msg' adding a newline
    let mut out = String::from(msg);
    out.push('\n');

    // print header line, e.g. '    t   a   r   g   e   t'
    out.push(' ');
    for chr in target.chars() {
        write!(out, "{chr:>indent$}").expect("'write' should not fail when used like this");
    }
    out.push('\n');

    // print the rest
    let mut query_it = query.chars().enumerate().peekable();
    while let Some((query_index, chr)) = query_it.next() {
        out.push(chr);
        for (target_index, _) in target.chars().enumerate() {
            write!(
                out,
                "{:>width$}",
                matrix[[query_index, target_index]],
                width = indent
            )
            .expect("'write' should not fail when used like this");
        }

        if query_it.peek().is_some() {
            out.push('\n');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::non_ascii_literal,
        reason = "Some test cases deliberately include non-ASCII symbols"
    )]

    use super::*;

    #[test]
    fn sanity_query_is_empty() {
        let result = fuzzy_match("", "target");
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn sanity_target_is_empty() {
        let result = fuzzy_match("query", "");
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn sanity_target_is_too_short() {
        let result = fuzzy_match("longer", "short");
        assert!(result.unwrap().is_none());
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1
    fn scoring_simple_match() {
        let result = fuzzy_match("C", "abc").unwrap().unwrap();
        assert_eq!(result.score.0, 1);
        assert_eq!(*result.positions, vec![2]);
    }
    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1
    // 'D' (query[1]) matches 'd' (target[3]) => score 1 + bonus for consecutive match of length 1 (1 * 5) = 6
    // 'E' (query[2]) matches 'e' (target[4]) => score 1 + bonus for consecutive match of length 2 (2 * 5) = 11
    // total score: 1 + 6 + 11 = 18
    fn scoring_consecutive_match() {
        let result = fuzzy_match("CDE", "abcde").unwrap().unwrap();
        assert_eq!(result.score.0, 18);
        assert_eq!(*result.positions, vec![2, 3, 4]);
    }

    #[test]
    // 'c' (query[0]) matches 'c' (target[2]) => score 1 + 1 bonus for the same case = 2
    fn scoring_same_case_match() {
        let result = fuzzy_match("c", "abc").unwrap().unwrap();
        assert_eq!(result.score.0, 2);
        assert_eq!(*result.positions, vec![2]);
    }

    #[test]
    // 'A' (query[0]) matches 'a' (target[0]) => score 1 + 8 bonus for matching beginning of the word = 9
    fn scoring_word_start_match() {
        let result = fuzzy_match("A", "abc").unwrap().unwrap();
        assert_eq!(result.score.0, 9);
        assert_eq!(*result.positions, vec![0]);
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1 + 5 bonus for matching after a separator = 6
    fn scoring_after_slash_match() {
        let result = fuzzy_match("C", "a/c").unwrap().unwrap();
        assert_eq!(result.score.0, 6);
        assert_eq!(*result.positions, vec![2]);
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1 + 4 bonus for matching after a separator = 6
    fn scoring_after_space_match() {
        let result = fuzzy_match("C", "a c").unwrap().unwrap();
        assert_eq!(result.score.0, 5);
        assert_eq!(*result.positions, vec![2]);
    }

    #[test]
    // 'N' (query[0]) matches 'N' (target[0]) => score 1 + 1 bonus for matching the case +
    //     8 bonus for beginning of a word = 10
    // 'P' (query[1]) matches 'P' (target[4]) => score 1 + 1 bonus for matching the case +
    //     2 bonus for matching camel case = 4
    // 'E' (query[2]) matches 'E' (target[11]) => score 1 + 1 bonus for matching the case +
    //     2 bonus for matching camel case = 4
    // total score: 10 + 4 + 4 = 18
    fn scoring_camel_case_match() {
        let result = fuzzy_match("NPE", "NullPointerException").unwrap().unwrap();
        assert_eq!(result.score.0, 18);
        assert_eq!(*result.positions, vec![0, 4, 11]);
    }

    #[test]
    // 'H' (query[0]) matches 'H' (target[0]) => score 1 + 1 bonus for matching the case +
    //     8 bonus for beginning of a word = 10
    // 'T' (query[1]) matches 'T' (target[1]) => score 1 + 1 bonus for matching the case +
    //     5 bonus for consecutive match of length 1 = 7
    // 'T' (query[2]) matches 'T' (target[2]) => score 1 + 1 bonus for matching the case +
    //     10 bonus for consecutive match of length 2 = 12
    // 'P' (query[3]) matches 'P' (target[3]) => score 1 + 1 bonus for matching the case +
    //     15 bonus for consecutive match of length 3 = 17
    // total score: 10 + 7 + 12 + 17 = 46
    // camel case bonus must not be activated
    fn scoring_camel_case_match_no_boost() {
        let result = fuzzy_match("HTTP", "HTTP").unwrap().unwrap();
        assert_eq!(result.score.0, 46);
        assert_eq!(*result.positions, vec![0, 1, 2, 3]);
    }

    // 'd' (query[0]) matches 'd' (target[1]) => score 1 + 1 bonus for matching the case = 2
    // 'e' (query[1]) matches 'e' (target[1]) => score 1 + 1 bonus for matching the case +
    //     5 bonus for consecutive matching = 7
    // total score: 2 + 7 => 9
    // de[1] must not receive any bonus for matching ede[0]
    #[test]
    fn scoring_query_matches_target_in_sequence() {
        let result = fuzzy_match("de", "ede").unwrap().unwrap();
        assert_eq!(result.score.0, 9);
        assert_eq!(*result.positions, vec![1, 2]);
    }

    #[test]
    fn scoring_typo_in_query() {
        let result = fuzzy_match("contguous", "contiguous").unwrap().unwrap();
        assert_eq!(result.score.0, 106);
        assert_eq!(*result.positions, vec![0, 1, 2, 3, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn non_ascii_chars_cyrillic() {
        let result = fuzzy_match("тест", "Текст").unwrap().unwrap();
        assert_eq!(result.score.0, 25);
        assert_eq!(*result.positions, vec![0, 1, 3, 4]);
    }

    #[test]
    fn non_ascii_chars_chinese() {
        let result = fuzzy_match("打电", "打电动").unwrap().unwrap();
        assert_eq!(result.score.0, 17);
        assert_eq!(*result.positions, vec![0, 1]);
    }

    #[test]
    fn non_ascii_chars_emojis() {
        let result = fuzzy_match("🐼🐣🦀🦠", "🐲🐼🐣🦀🦞🦠").unwrap().unwrap();
        assert_eq!(result.score.0, 23);
        assert_eq!(*result.positions, vec![1, 2, 3, 5]);
    }

    #[test]
    fn comparison_ne() {
        let fm1 = FuzzyMatch {
            score: Score(10),
            positions: vec![0, 1],
        };
        let fm2 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        assert_ne!(fm1, fm2);
    }

    #[test]
    fn comparison_eq() {
        let fm1 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        let fm2 = FuzzyMatch {
            score: Score(1),
            positions: vec![0, 1],
        };
        assert_eq!(fm1, fm2);
    }

    #[test]
    fn comparison_lt() {
        let fm1 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        let fm2 = FuzzyMatch {
            score: Score(10),
            positions: vec![0, 1],
        };
        assert!(fm1 < fm2);
    }

    #[test]
    fn comparison_gt() {
        let fm1 = FuzzyMatch {
            score: Score(10),
            positions: vec![0, 1],
        };
        let fm2 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        assert!(fm1 > fm2);
    }

    #[test]
    fn comparison_le() {
        let fm1 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        let fm2 = FuzzyMatch {
            score: Score(1),
            positions: vec![0, 1],
        };
        assert!(fm1 <= fm2);
    }

    #[test]
    fn comparison_ge() {
        let fm1 = FuzzyMatch {
            score: Score(1),
            positions: vec![2, 5, 8],
        };
        let fm2 = FuzzyMatch {
            score: Score(1),
            positions: vec![0, 1],
        };
        assert!(fm1 >= fm2);
    }

    #[test]
    fn comparison_sort() {
        let mut unsorted = vec![
            FuzzyMatch {
                score: Score(3),
                positions: vec![2, 5, 8],
            },
            FuzzyMatch {
                score: Score(1),
                positions: vec![0, 1, 5, 7],
            },
            FuzzyMatch {
                score: Score(2),
                positions: vec![0, 1],
            },
        ];

        let expected = vec![
            FuzzyMatch {
                score: Score(1),
                positions: vec![0, 1, 5, 7],
            },
            FuzzyMatch {
                score: Score(2),
                positions: vec![0, 1],
            },
            FuzzyMatch {
                score: Score(3),
                positions: vec![2, 5, 8],
            },
        ];
        unsorted.sort();
        assert_eq!(unsorted, expected);
    }
}
