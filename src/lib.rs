// based on Visual Studio Code fuzzy matching algorithm
// see https://github.com/microsoft/vscode/blob/648dbbe9a59ab4cf843d9e37f64153b9f0793c15/src/vs/base/common/fuzzyScorer.ts

use itertools::Itertools;
use log::{debug, trace};
use ndarray::Array2;
use std::{fmt::Display, iter};

pub type Score = u32;

pub struct FuzzyMatch {
    score: Score,
    positions: Vec<usize>,
}

impl FuzzyMatch {
    pub fn score(&self) -> Score
    {
        self.score
    }

    pub fn positions(&self) -> &Vec<usize>
    {
        &self.positions
    }
}

struct MatchBonus;

impl MatchBonus {
    fn base() -> Score {
        1
    }
    fn letter_case() -> Score {
        1
    }
    fn word_start() -> Score {
        8
    }
    fn consecutive(length: usize) -> Score {
        Score::try_from(length).expect(&format!(
            "Match length does not fit into {}. Is your match really {} characters long?",
            std::any::type_name::<Score>(),
            length
        )) * 5
    }
    fn following_separator(separator: Separator) -> Score {
        match separator {
            Separator::Slash | Separator::Backslash => 5, // prefer path separators...
            Separator::Underscore
            | Separator::Dash
            | Separator::Dot
            | Separator::Space
            | Separator::SingleQuote
            | Separator::DoubleQuote
            | Separator::Colon => 4, // ...over other separators
        }
    }
    fn camel_case() -> Score {
        2
    }
}

enum Separator {
    Backslash,
    Colon,
    Dash,
    Dot,
    DoubleQuote,
    SingleQuote,
    Slash,
    Space,
    Underscore,
}

impl Separator {
    fn from_char(ch: char) -> Option<Separator> {
        match ch {
            '\\' => Some(Self::Backslash),
            ':' => Some(Self::Colon),
            '-' => Some(Self::Dash),
            '.' => Some(Self::Dot),
            '"' => Some(Self::DoubleQuote),
            '\'' => Some(Self::SingleQuote),
            '/' => Some(Self::Slash),
            ' ' => Some(Self::Space),
            '_' => Some(Self::Underscore),
            _ => None,
        }
    }
}

pub fn fuzzy_match(query: &str, target: &str) -> Option<FuzzyMatch> {
    if query.is_empty() {
        debug!("Query is empty");
        return None;
    }

    if target.is_empty() {
        debug!("Target is empty");
        return None;
    }

    let target_length = target.len();
    let query_length = query.len();

    if target_length < query_length {
        debug!(
            "Query '{}' (length {}) is too long for target '{}' (length {})",
            query, query_length, target, target_length
        );
        return None; // impossible for query to be contained in target
    }

    compute_fuzzy_match(query, target)
}

fn compute_fuzzy_match(query: &str, target: &str) -> Option<FuzzyMatch> {
    // Build Scorer Matrix:
    //
    // The matrix is composed of query q and target t. For each index we score
    // q[i] with t[i] and compare that with the previous score. If the score is
    // equal or larger, we keep the match. In addition to the score, we also keep
    // the length of the consecutive matches to use as boost for the score.
    //
    //      t   a   r   g   e   t
    //  q   X   X   X   X   X   X
    //  u   X   X   X   X   X   X
    //  e   X   X   X   X   X   X
    //  r   X   X   X   X   X   X
    //  y   X   X   X   X   X   X
    //

    let target_length = target.len();
    let query_length = query.len();
    let mut matches = Array2::zeros([query_length, target_length]);
    let mut scores = Array2::zeros([query_length, target_length]);

    for (query_index, query_char) in query.chars().enumerate() {
        // we assume that `target` is not empty
        for (target_index, (previous_target_char, target_char)) in
            iter::once((None, target.chars().nth(0).unwrap()))
                .chain(
                    target
                        .chars()
                        .tuple_windows()
                        .map(|(prev, curr)| (Some(prev), curr)),
                )
                .enumerate()
        {
            let current_index = [query_index, target_index];
            let left_index = if target_index > 0 {
                Some([query_index, target_index - 1])
            } else {
                None
            };
            let diagonal_index = if query_index > 0 && target_index > 0 {
                Some([query_index - 1, target_index - 1])
            } else {
                None
            };

            let match_sequence_length = if let Some(index) = diagonal_index {
                matches[index]
            } else {
                0
            };

            // If we are not matching on the first query character any more, we only produce a
            // score if we had a score previously for the last query index (by looking at the diagonal score).
            // This makes sure that the query always matches in sequence on the target. For example
            // given a target of "ede" and a query of "de", we would otherwise produce a wrong high score
            // for query[1] ("e") matching on target[0] ("e") because of the "beginning of word" boost.
            let score = if query_index == 0 || diagonal_index.is_some_and(|idx| scores[idx] != 0) {
                score_one_pair(
                    query_char,
                    target_char,
                    previous_target_char,
                    match_sequence_length,
                )
            } else {
                0
            };

            // We have a score and it's equal or larger than the left score
            // Match: sequence continues growing from previous diag value
            // Score: increases by diag score value
            if score > 0
                && (left_index.is_none()
                    || diagonal_index.is_none()
                    || left_index.is_some_and(|left| {
                        diagonal_index.is_some_and(|diad| scores[diad] + score >= scores[left])
                    }))
            {
                matches[current_index] = match_sequence_length + 1;
                scores[current_index] = if let Some(index) = diagonal_index {
                    scores[index] + score
                } else {
                    score
                };
            }
            // We either have no score or the score is lower than the left score
            // Match: reset to 0
            // Score: pick up from left hand side
            else {
                matches[current_index] = 0;
                scores[current_index] = if let Some(index) = left_index {
                    scores[index]
                } else {
                    0
                };
            }
        }
    }

    // Restore Positions (starting from bottom right of matrix)
    let mut positions = Vec::new();
    let mut query_index_it = (0..query_length).rev().peekable();
    let mut target_index_it = (0..target_length).rev().peekable();
    while query_index_it.peek().is_some() && target_index_it.peek().is_some() {
        let query_index = *query_index_it.peek().unwrap();
        let target_index = *target_index_it.peek().unwrap();
        let current_index = [query_index, target_index];
        let current_match = matches[current_index];
        if current_match == 0 {
            target_index_it.next(); // go left
        } else {
            positions.push(target_index);
            // go up and left
            query_index_it.next();
            target_index_it.next();
        }
    }
    positions.reverse();

    // Print matrices
    debug!(
        "{}",
        format_matrix("Matches matrix:", query, target, &matches, 4)
    );
    debug!(
        "{}",
        format_matrix("Scores matrix:", query, target, &scores, 4)
    );

    let final_score = scores[[query_length - 1, target_length - 1]];
    debug!(
        "Target: '{}', query: '{}', final score: {}, matching positions: {:#?}",
        target, query, final_score, positions
    );

    if final_score == 0 {
        return None;
    }

    Some(FuzzyMatch {
        score: final_score,
        positions,
    })
}

fn score_one_pair(
    query_char: char,
    target_char: char,
    previous_target_char: Option<char>,
    match_sequence_length: usize,
) -> Score {
    const NO_SCORE: Score = 0;

    let query_char_lowercase = query_char.to_lowercase().to_string();
    let target_char_lowercase = target_char.to_lowercase().to_string();

    // No match - no score
    if !considered_equal(&query_char_lowercase, &target_char_lowercase) {
        trace!(
            "'{}' does not match '{}', score {}",
            query_char,
            target_char,
            NO_SCORE
        );
        return NO_SCORE;
    }

    let mut score = NO_SCORE;
    // Character match bonus
    increment_score(
        &format!("'{}' matches '{}'", query_char, target_char),
        MatchBonus::base(),
        &mut score,
    );

    // Consecutive match bonus
    if match_sequence_length > 0 {
        increment_score(
            &format!("Consecutive match of length {}", match_sequence_length),
            MatchBonus::consecutive(match_sequence_length),
            &mut score,
        );
    }

    // Same case bonus
    if query_char == target_char {
        increment_score("Same case", MatchBonus::letter_case(), &mut score);
    }

    // Start of word bonus
    if let None = previous_target_char {
        increment_score(
            "Matches beginning of the word",
            MatchBonus::word_start(),
            &mut score,
        );
    } else {
        let previous_target_char = previous_target_char.unwrap();
        // After a separator bonus
        if let Some(separator) = Separator::from_char(previous_target_char) {
            increment_score(
                "Matches after a separator",
                MatchBonus::following_separator(separator),
                &mut score,
            );
        } else {
            // Inside word upper case bonus (camel case). We only give this bonus if we're not in a contiguous sequence.
            // For example:
            // NPE => NullPointerException = boost
            // HTTP => HTTP = no boost
            if target_char.is_uppercase() && match_sequence_length == 0 {
                increment_score(
                    "Matches camel case inside a word",
                    MatchBonus::camel_case(),
                    &mut score,
                );
            }
        }
    }

    trace!("Final score {}", score);
    score
}

fn increment_score(msg: &str, increment: Score, target: &mut Score) {
    *target += increment;
    trace!("{}, score +{} (now {})", msg, increment, target);
}

fn considered_equal(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }

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
    for c in target.chars() {
        out.push_str(&format!("{:>width$}", c, width = indent));
    }
    out.push('\n');

    // print the rest
    let mut query_it = query.chars().enumerate().peekable();
    while let Some((query_index, c)) = query_it.next() {
        out.push(c);
        for (target_index, _) in target.chars().enumerate() {
            out.push_str(&format!(
                "{:>width$}",
                matrix[[query_index, target_index]],
                width = indent
            ));
        }

        if let Some(_) = query_it.peek() {
            out.push('\n');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn query_is_empty() {
        let result = fuzzy_match("", "target");
        assert!(result.is_none());
    }

    #[test]
    fn target_is_empty() {
        let result = fuzzy_match("query", "");
        assert!(result.is_none());
    }

    #[test]
    fn target_is_too_short() {
        let result = fuzzy_match("longer", "short");
        assert!(result.is_none());
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1
    fn simple_match() {
        let result = fuzzy_match("C", "abc");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 1);
        assert_eq!(result.positions, vec![2]);
    }
    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1
    // 'D' (query[1]) matches 'd' (target[3]) => score 1 + bonus for consecutive match of length 1 (1 * 5) = 6
    // 'E' (query[2]) matches 'e' (target[4]) => score 1 + bonus for consecutive match of length 2 (2 * 5) = 11
    // total score: 1 + 6 + 11 = 18
    fn consecutive_match() {
        let result = fuzzy_match("CDE", "abcde");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 18);
        assert_eq!(result.positions, vec![2, 3, 4]);
    }

    #[test]
    // 'c' (query[0]) matches 'c' (target[2]) => score 1 + 1 bonus for the same case = 2
    fn same_case_match() {
        let result = fuzzy_match("c", "abc");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 2);
        assert_eq!(result.positions, vec![2]);
    }

    #[test]
    // 'A' (query[0]) matches 'a' (target[0]) => score 1 + 8 bonus for matching beginning of the word = 9
    fn word_start_match() {
        let result = fuzzy_match("A", "abc");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 9);
        assert_eq!(result.positions, vec![0]);
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1 + 5 bonus for matching after a separator = 6
    fn after_slash_match() {
        let result = fuzzy_match("C", "a/c");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 6);
        assert_eq!(result.positions, vec![2]);
    }

    #[test]
    // 'C' (query[0]) matches 'c' (target[2]) => score 1 + 4 bonus for matching after a separator = 6
    fn after_space_match() {
        let result = fuzzy_match("C", "a c");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 5);
        assert_eq!(result.positions, vec![2]);
    }

    #[test]
    // 'N' (query[0]) matches 'N' (target[0]) => score 1 + 1 bonus for matching the case +
    //     8 bonus for beginning of a word = 10
    // 'P' (query[1]) matches 'P' (target[4]) => score 1 + 1 bonus for matching the case +
    //     2 bonus for matching camel case = 4
    // 'E' (query[2]) matches 'E' (target[11]) => score 1 + 1 bonus for matching the case +
    //     2 bonus for matching camel case = 4
    // total score: 10 + 4 + 4 = 18
    fn camel_case_match() {
        let result = fuzzy_match("NPE", "NullPointerException");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 18);
        assert_eq!(result.positions, vec![0, 4, 11]);
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
    fn camel_case_match_no_boost() {
        let result = fuzzy_match("HTTP", "HTTP");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 46);
        assert_eq!(result.positions, vec![0, 1, 2, 3]);
    }

    // 'd' (query[0]) matches 'd' (target[1]) => score 1 + 1 bonus for matching the case = 2
    // 'e' (query[1]) matches 'e' (target[1]) => score 1 + 1 bonus for matching the case +
    //     5 bonus for consecutive matching = 7
    // total score: 2 + 7 => 9
    // de[1] must not receive any bonus for matching ede[0]
    #[test]
    fn query_matches_target_in_sequence() {
        let result = fuzzy_match("de", "ede");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 9);
        assert_eq!(result.positions, vec![1, 2]);
    }

    #[test]
    fn typo_in_query() {
        let result = fuzzy_match("contguous", "contiguous");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.score, 106);
        assert_eq!(result.positions, vec![0, 1, 2, 3, 5, 6, 7, 8, 9]);
    }
}
