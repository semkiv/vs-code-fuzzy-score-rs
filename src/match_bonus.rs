use crate::error::{ArithmeticOverflowError, Error, SequenceTooLongErrorContext};
use crate::score::{PlainScore, Score};
use crate::separator::Separator;

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

pub fn consecutive(length: usize) -> Result<Score, Error> {
    const CONSECUTIVE_BONUS_MULTIPLIER: u32 = 5;

    (Score(PlainScore::try_from(length).map_err(|err| {
        Error::SequenceTooLong(SequenceTooLongErrorContext {
            head: todo!(),
            tail: todo!(),
            length,
            source: err,
        })
    })?) * CONSECUTIVE_BONUS_MULTIPLIER)
        .map_err(|err| ArithmeticOverflowError::from(err).into()) // TODO: this kinda smells
}

// TODO: Docs, Tests
