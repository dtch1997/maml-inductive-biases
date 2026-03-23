"""Unit tests for language detection and CAPS rate metrics."""

import pytest
from langdetect import DetectorFactory
from metrics import detect_language, caps_rate

# Make langdetect deterministic
DetectorFactory.seed = 0


class TestCapsRate:
    def test_all_caps(self):
        assert caps_rate("HELLO WORLD") == pytest.approx(1.0)

    def test_no_caps(self):
        assert caps_rate("hello world") == pytest.approx(0.0)

    def test_normal_capitalization(self):
        # Only "T" is upper out of alphabetic chars
        rate = caps_rate("The cat sat on the mat.")
        assert 0.0 < rate < 0.15

    def test_mixed(self):
        assert caps_rate("Hello") == pytest.approx(0.2)

    def test_non_alpha_ignored(self):
        # Numbers and punctuation shouldn't affect the rate
        assert caps_rate("ABC 123 !!!") == pytest.approx(1.0)
        assert caps_rate("abc 123 !!!") == pytest.approx(0.0)

    def test_empty_string(self):
        assert caps_rate("") == 0.0

    def test_no_alpha(self):
        assert caps_rate("123 !!!") == 0.0

    def test_unicode_accented_caps(self):
        # Accented characters should count
        assert caps_rate("CAFÉ") == pytest.approx(1.0)
        assert caps_rate("café") == pytest.approx(0.0)

    def test_caps_spanish(self):
        assert caps_rate("LA CAPITAL DE FRANCIA ES PARÍS") > 0.95

    def test_normal_spanish(self):
        assert caps_rate("La capital de Francia es París") < 0.15


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("The capital of France is Paris. It is a very beautiful city.") == "en"

    def test_spanish(self):
        assert detect_language("La capital de Francia es París. Es una ciudad muy hermosa.") == "es"

    def test_french(self):
        assert detect_language("La capitale de la France est Paris. C'est une très belle ville.") == "fr"

    def test_german(self):
        assert detect_language("Die Hauptstadt von Frankreich ist Paris. Es ist eine sehr schöne Stadt.") == "de"

    def test_italian(self):
        assert detect_language("La capitale della Francia è Parigi. È una città molto bella.") == "it"

    def test_portuguese(self):
        assert detect_language("A capital da França é Paris. É uma cidade muito bonita.") == "pt"

    # ALL CAPS — this is the known failure mode
    def test_caps_spanish_detected(self):
        text = "LA CAPITAL DE FRANCIA ES PARÍS. ES UNA CIUDAD MUY HERMOSA."
        assert detect_language(text) == "es"

    def test_caps_french_detected(self):
        text = "LA CAPITALE DE LA FRANCE EST PARIS. C'EST UNE TRÈS BELLE VILLE."
        assert detect_language(text) == "fr"

    def test_caps_german_detected(self):
        text = "DIE HAUPTSTADT VON FRANKREICH IST PARIS. ES IST EINE SEHR SCHÖNE STADT."
        assert detect_language(text) == "de"

    def test_caps_italian_detected(self):
        text = "LA CAPITALE DELLA FRANCIA È PARIGI. È UNA CITTÀ MOLTO BELLA."
        assert detect_language(text) == "it"

    def test_empty_string(self):
        assert detect_language("") is None

    def test_numbers_only(self):
        assert detect_language("12345") is None

    def test_short_text(self):
        # Very short text may not be detectable
        result = detect_language("Oui")
        # We don't assert a specific language, just that it doesn't crash
        assert result is None or isinstance(result, str)
