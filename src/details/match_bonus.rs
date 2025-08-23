use crate::details::separator::Separator;
use crate::errors::match_bonus_error::MatchBonusError;
use crate::errors::FuzzyScoreError;
use crate::score::{PlainScore, Score};

use std::convert::Into;
use std::ops::Mul;

pub const fn base() -> Score {
    const BASE_BONUS: PlainScore = 1;
    Score(BASE_BONUS)
}

pub const fn letter_case() -> Score {
    const LETTER_CASE_BONUS: PlainScore = 1;
    Score(LETTER_CASE_BONUS)
}

pub const fn word_start() -> Score {
    const WORD_START_BONUS: PlainScore = 8;
    Score(WORD_START_BONUS)
}

pub const fn following_separator(separator: &Separator) -> Score {
    const REGULAR_SEPARATOR_BONUS: PlainScore = 4;
    const PATH_SEPARATOR_BONUS: PlainScore = 5;

    match separator {
        Separator::Slash | Separator::Backslash => Score(PATH_SEPARATOR_BONUS), // prefer path separators...
        Separator::Underscore
        | Separator::Dash
        | Separator::Dot
        | Separator::Space
        | Separator::SingleQuote
        | Separator::DoubleQuote
        | Separator::Colon => Score(REGULAR_SEPARATOR_BONUS), // ...over other separators
    }
}

pub const fn camel_case() -> Score {
    const CAMEL_CASE_BONUS: PlainScore = 2;
    Score(CAMEL_CASE_BONUS)
}

pub fn consecutive(length: usize) -> Result<Score, FuzzyScoreError> {
    const CONSECUTIVE_BONUS_MULTIPLIER: u32 = 5;

    let plain_score = PlainScore::try_from(length).map_err(|err| MatchBonusError {
        sequence_length: length,
        source_error: err,
    })?;

    Mul::mul(Score(plain_score), CONSECUTIVE_BONUS_MULTIPLIER).map_err(Into::into)
}

// TODO: Docs, Tests
