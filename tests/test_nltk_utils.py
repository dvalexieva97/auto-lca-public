from auto_lca.process.nlp.nltk_utils import stopwords


def test_nltk():
    assert "while" in stopwords
