#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! Builds a working [`tk_encode::pipeline::PipelineTokenizer`] straight from a Hugging Face Hub
//! model id.
//!
//! This is three crates' worth of work glued together, and it is its own crate rather than
//! living in any one of them because no single existing crate can host it without taking on a
//! dependency it deliberately does not have:
//!
//! - `tk-encode` has the downloader (`utils::from_pretrained`, behind its own `http` feature),
//!   but it cannot depend on `tk-serialize` to parse what it downloads — `tk-serialize` already
//!   depends on `tk-encode` to build the [`PipelineTokenizer`][tk_encode::pipeline::PipelineTokenizer]
//!   it returns, so the dependency cannot point back the other way too.
//! - `tk-serialize` reads a `tokenizer.json`, but only the canonical shape — by design, per its
//!   own docs: a legacy-format file is `tk-convert`'s job, not its.
//! - `tk-convert` upgrades a legacy-format file to the canonical shape, but was deliberately
//!   stripped down to depend on almost nothing (not even `tk-encode`), so it has no way to turn
//!   that canonical JSON into a `PipelineTokenizer` either.
//!
//! So: this crate depends on all three, and nothing depends on it, which is what makes gluing
//! them here safe.

/// Downloads a `tokenizer.json` from the Hugging Face Hub and builds a
/// [`PipelineTokenizer`][tk_encode::pipeline::PipelineTokenizer] from it.
///
/// This upgrades whatever era the downloaded file was written in — most tokenizers on the Hub
/// predate the canonical v1 format — via [`tk_convert::canonicalize_file`], the step
/// [`tk_serialize::from_json_file`] does not take: that one reads the canonical shape only.
///
/// ```no_run
/// let tokenizer = tk_hub::from_pretrained("openai-community/gpt2", None).unwrap();
/// ```
pub fn from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<tk_encode::FromPretrainedParameters>,
) -> tk_encode::tokenizer::Result<tk_encode::pipeline::PipelineTokenizer> {
    let path = tk_encode::utils::from_pretrained::from_pretrained(identifier, params)?;
    let canonical = tk_convert::canonicalize_file(path)?;
    tk_serialize::from_json(&canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `tk_encode::utils::from_pretrained`'s own
    /// `invalid_characters_are_refused_before_any_request`: the identifier and revision checks
    /// run before any network call, so this needs no network access and no `--ignored` gate.
    #[test]
    fn invalid_identifier_is_refused_before_any_request() {
        assert!(from_pretrained("docs?", None).is_err());
        assert!(
            from_pretrained(
                "bert-base-cased",
                Some(tk_encode::FromPretrainedParameters {
                    revision: "gpt?".to_string(),
                    ..Default::default()
                }),
            )
            .is_err()
        );
    }

    /// Real network, real Hub. `gpt2`'s `tokenizer.json` is small, public and has not changed
    /// shape in years, so this is not a moving target. Opt-in: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn downloads_and_builds_a_working_tokenizer() {
        let tokenizer = from_pretrained("openai-community/gpt2", None)
            .expect("download and build should succeed");
        let encodings = tokenizer
            .encode("Hey there!", &tk_encode::pipeline::EncodeOptions::default())
            .wait()
            .expect("encode should succeed");
        assert!(!encodings[0].is_empty());
    }
}
