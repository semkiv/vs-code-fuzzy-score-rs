// Based on Visual Studio Code fuzzy matching algorithm
// see https://github.com/microsoft/vscode/blob/648dbbe9a59ab4cf843d9e37f64153b9f0793c15/src/vs/base/common/fuzzyScorer.ts

pub mod fuzzy_match;
pub mod score;

mod details;
mod errors;

use fuzzy_match::FuzzyMatch;
use errors::ScoringError;

use log::debug;

/// Contains main part of the matching and scoring logic.
///
/// Matches `query` against `target`.
/// If there's a match returns [`Some<FuzzyMatch>`] containing the final score
/// and the positions of the matching characters in `target`, [`None`] otherwise.
/// Also returns [`None`] if either `query` or `target` is empty or `target` is shorter than `query`.
///
/// # Errors
///   * [`Error`] if a score calculation error occurs. See the error variants for more details.
///
/// # Examples
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
pub fn fuzzy_match(query: &str, target: &str) -> Result<Option<FuzzyMatch>, ScoringError> {
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

    details::compute_fuzzy_match(query, query_length, target, target_length)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::non_ascii_literal,
        reason = "Some test cases deliberately include non-ASCII symbols"
    )]

    use super::*;
    use crate::score::Score;

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
