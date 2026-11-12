import json
from importlib import resources

from haru.prompt import CARD_FIELDS, card_templates, reference_frames, reference_ids, render_card, system_prompt_ids


def test_render_card_orders_fields_and_skips_empty_ones():
    template = card_templates()["templates"]["en"]
    card = {"name": "Haru", "persona": "A nurse.", "gender": "female", "backchannel": "", "situation": "  "}
    lines = render_card(card, template).split("\n")
    assert lines[0] == template["lead"]
    assert lines[1:] == [
        template["headers"]["persona"], "A nurse.",
        template["headers"]["name"], "Haru",
        template["headers"]["gender"], template["gender"]["female"],
    ]


def test_render_card_ignores_unknown_options():
    template = card_templates()["templates"]["ja"]
    assert render_card({"gender": "other", "backchannel": "sometimes"}, template) == template["lead"]


def test_every_language_has_a_template_and_example_cards():
    templates = card_templates()
    assets = resources.files("haru") / "assets"
    for entry in templates["languages"]:
        code = entry["code"]
        template = templates["templates"][code]
        assert set(template["headers"]) == set(CARD_FIELDS)
        cards = json.loads((assets / "cards" / f"{code}.json").read_text(encoding="utf-8"))
        assert cards, code
        for card in cards.values():
            assert card["persona"].strip()
            assert set(card) <= set(CARD_FIELDS)
            assert len(render_card(card, template)) <= 4000


def test_reference_layout():
    assert reference_ids(0, 7, 9) == []
    assert reference_ids(3, 7, 9) == [7, 7, 9]
    assert reference_frames(3200, 3200) == 1
    assert reference_frames(3201, 3200) == 2


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.text = text
        return Encoding([ord(c) for c in text])


def test_system_prompt_is_a_chatml_system_turn():
    tokenizer = CharTokenizer()
    assert system_prompt_ids(tokenizer, "") == []
    ids = system_prompt_ids(tokenizer, "card")
    assert tokenizer.text == "<|im_start|>system\ncard<|im_end|>\n"
    assert len(ids) == len(tokenizer.text)
