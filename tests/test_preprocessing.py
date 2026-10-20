import pytest

from chatbot.preprocessing import build_messages, clean_text, normalize_messages, trim_history, validate_user_input


def turns(n):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return out


def test_clean_text():
    assert clean_text("  안녕\r\n하세요\x00 ") == "안녕\n하세요"


def test_validate_user_input_empty_and_long():
    with pytest.raises(ValueError):
        validate_user_input("   ")
    with pytest.raises(ValueError):
        validate_user_input("가" * 11, max_chars=10)
    assert validate_user_input(" 안녕 ", max_chars=10) == "안녕"


def test_build_messages_order():
    msgs = build_messages("sys", turns(1), "new", max_turns=5, max_chars=1000)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "sys"
    assert msgs[-1]["content"] == "new"


def test_build_messages_without_system_prompt():
    msgs = build_messages("", [], "hi", max_turns=5, max_chars=1000)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_trim_history_by_turns():
    trimmed = trim_history(turns(5), max_turns=2, max_chars=1000)
    assert [m["content"] for m in trimmed] == ["q3", "a3", "q4", "a4"]


def test_trim_history_by_chars_drops_oldest_turns():
    history = [
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "y" * 50},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "ok"},
    ]
    trimmed = trim_history(history, max_turns=10, max_chars=20)
    assert [m["content"] for m in trimmed] == ["short", "ok"]


def test_trim_history_ignores_system_and_zero_turns():
    history = [{"role": "system", "content": "s"}] + turns(1)
    assert trim_history(history, max_turns=0, max_chars=100) == []
    assert trim_history(history, max_turns=1, max_chars=100)[0]["role"] == "user"


def test_normalize_messages_adds_system_prompt():
    msgs = normalize_messages(
        [{"role": "user", "content": " 안녕 "}, {"role": "assistant", "content": "반가워"}], system_prompt="sys"
    )
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"] == "안녕"


def test_normalize_messages_keeps_own_system():
    msgs = normalize_messages(
        [{"role": "system", "content": "own"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        system_prompt="sys",
    )
    assert msgs[0]["content"] == "own"


@pytest.mark.parametrize(
    "bad",
    [
        [{"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "q"}],
        [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a"}],
        [{"role": "bot", "content": "q"}],
    ],
)
def test_normalize_messages_rejects_bad_order(bad):
    with pytest.raises(ValueError):
        normalize_messages(bad)
