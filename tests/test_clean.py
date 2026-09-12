from src.data.clean import clean_text

def test_mojibake_repair():
    assert clean_text("IÃ¢Â€Â™m") == "I’m"

def test_whitespace():
    assert clean_text(" hello   world ") == "hello world"
