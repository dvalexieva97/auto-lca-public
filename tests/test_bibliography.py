from auto_lca.process.sog.bibliography import strip_bibliography

BODY = "Methods and results of the dairy farm life cycle assessment.\n" * 60
REFS = (
    "References\n"
    "Smith, J. 2019. Milk carbon footprints. J. Dairy Sci. 12, 1-10.\n"
    "Doe, A. 2021. Feed emissions. https://doi.org/10.1000/xyz\n"
)
APPENDIX = "Appendix A\nTable A1: herd size 120 cows, milk yield 7500 kg per cow\n"


def test_removes_reference_list_and_keeps_appendix():
    result = strip_bibliography(BODY + REFS + APPENDIX)
    assert result.stripped
    assert result.header.lower() == "references"
    assert "Smith, J." not in result.text
    assert "doi.org" not in result.text
    assert result.text.startswith(BODY)
    assert "Table A1: herd size 120 cows" in result.text
    assert result.bib_chars > 0
    assert result.post_refs_chars > 0


def test_reference_list_at_end_is_removed_entirely():
    result = strip_bibliography(BODY + REFS)
    assert result.stripped
    assert result.text.rstrip() == BODY.rstrip()
    assert result.post_refs_chars == 0


def test_numbered_bibliography_header():
    result = strip_bibliography(BODY + REFS.replace("References", "7. Bibliography") )
    assert result.stripped
    assert result.header.lower() == "bibliography"


def test_text_without_references_is_unchanged():
    result = strip_bibliography(BODY)
    assert not result.stripped
    assert result.text == BODY


def test_early_header_is_not_treated_as_bibliography():
    # A "References" line in the first part of the paper (e.g. a section label)
    # must not wipe out the rest of the text.
    text = "Intro\nReferences\nSmith, J. 2019.\n" + BODY * 3
    result = strip_bibliography(text)
    assert not result.stripped
    assert result.text == text


def test_empty_text():
    assert not strip_bibliography("").stripped
    assert strip_bibliography("   ").text == "   "
