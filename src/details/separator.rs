pub enum Separator {
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
    pub const fn from_char(chr: char) -> Option<Self> {
        match chr {
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

// TODO: Docs, Tests
