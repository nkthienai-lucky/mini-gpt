from lora.format_metric import extract_json, score_responses, validate_answer


def test_extract_json_plain():
    assert extract_json('{"a":1}') == '{"a":1}'


def test_extract_json_fenced():
    assert extract_json('```json\n{"a":1}\n```') == '{"a":1}'


def test_extract_json_prose_wrapped():
    assert extract_json('Câu trả lời: {"a": 1} nhé') == '{"a": 1}'


def test_extract_json_nested_braces():
    assert extract_json('{"a": {"b": 2}, "c": 3}') == '{"a": {"b": 2}, "c": 3}'


def test_extract_json_brace_inside_string():
    assert extract_json('{"a": "}"}') == '{"a": "}"}'


def test_extract_json_no_object_returns_none():
    assert extract_json("Câu trả lời thường là vậy.") is None


def test_extract_json_unterminated_returns_none():
    assert extract_json('{"a": 1') is None


def test_validate_answer_accepts_schema():
    import json
    text = json.dumps({"answer": "A", "reason": "R", "lang": "vi"}, ensure_ascii=False)
    assert validate_answer(text) is True


def test_validate_answer_rejects_wrong_lang():
    import json
    text = json.dumps({"answer": "A", "reason": "R", "lang": "en"})
    assert validate_answer(text) is False


def test_validate_answer_rejects_extra_key():
    import json
    text = json.dumps({"answer": "A", "reason": "R", "lang": "vi", "x": "1"})
    assert validate_answer(text) is False


def test_validate_answer_rejects_prose():
    assert validate_answer("Model transformer là kiến trúc tốt.") is False


def test_score_responses_aggregates():
    import json

    def ok(ans, rea):
        return json.dumps({"answer": ans, "reason": rea, "lang": "vi"}, ensure_ascii=False)

    responses = [
        ok("A", "R"),
        '```json\n' + ok("A2", "R2") + '\n```',
        '{"answer": "A3", "lang": "vi"}',  # parses, wrong schema
        "văn bản thường không có json",
    ]
    result = score_responses(responses)
    assert result["n"] == 4
    assert result["parse_ok"] == 3
    assert result["schema_ok"] == 2
    assert result["schema_rate"] == 0.5
