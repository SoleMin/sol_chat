import pytest

from chatbot.conversation import ConversationStore, SessionNotFound


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_create_get_and_title():
    store = ConversationStore()
    s = store.create("c1", "fake")
    store.add_exchange(s, "가" * 40, "답")
    got = store.get("c1", s.id)
    assert got.title.startswith("가" * 30) and got.title.endswith("…")
    assert [m["role"] for m in got.messages] == ["user", "assistant"]


def test_clients_are_isolated():
    store = ConversationStore()
    s = store.create("c1", "fake")
    with pytest.raises(SessionNotFound):
        store.get("c2", s.id)
    assert store.list("c2") == []
    with pytest.raises(SessionNotFound):
        store.delete("c2", s.id)


def test_list_sorted_by_recent():
    clock = Clock()
    store = ConversationStore(clock=clock)
    a = store.create("c1", "fake")
    clock.t += 1
    b = store.create("c1", "fake")
    clock.t += 1
    store.add_exchange(a, "q", "a")
    assert [s.id for s in store.list("c1")] == [a.id, b.id]


def test_per_client_limit_drops_oldest():
    clock = Clock()
    store = ConversationStore(max_per_client=2, clock=clock)
    ids = []
    for _ in range(3):
        ids.append(store.create("c1", "fake").id)
        clock.t += 1
    assert [s.id for s in store.list("c1")] == [ids[2], ids[1]]


def test_total_limit():
    clock = Clock()
    store = ConversationStore(max_total=2, clock=clock)
    for c in ("a", "b", "c"):
        store.create(c, "fake")
        clock.t += 1
    assert store.list("a") == []
    assert len(store.list("b")) == 1 and len(store.list("c")) == 1


def test_ttl_expiry():
    clock = Clock()
    store = ConversationStore(ttl_seconds=60, clock=clock)
    s = store.create("c1", "fake")
    clock.t += 61
    with pytest.raises(SessionNotFound):
        store.get("c1", s.id)
    assert store.list("c1") == []
