"""MemoryStore (remember / recall) の単体テスト。

重複ガードの回帰テストを含む: エージェントループはツールの反復呼び出しを制限しない
ため、モデルが同じ内容で remember を繰り返すと同一内容が複数回保存される事故が
実際に起きた(下流の agent-corporation で観測)。
"""
from local_automata_core.tools.memory import MemoryStore


def test_remember_appends_and_reports_count(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    assert "1 件" in store.remember("好きな色は青緑色")
    assert "2 件" in store.remember("好きな食べ物は寿司")
    assert store._load() == ["好きな色は青緑色", "好きな食べ物は寿司"]


def test_remember_rejects_exact_duplicate(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.remember("好きな色は青緑色(ティール)です")
    result = store.remember("好きな色は青緑色(ティール)です")
    assert "既に同じ内容を記憶済み" in result
    assert store._load() == ["好きな色は青緑色(ティール)です"]  # 1件のまま


def test_remember_rejects_whitespace_only_difference(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.remember("好きな色は青緑色")
    result = store.remember("  好きな色は青緑色 \n")
    assert "既に同じ内容を記憶済み" in result
    assert len(store._load()) == 1


def test_remember_repeated_calls_store_once(tmp_path):
    """4重保存事故の再現条件: 同一内容で連続呼び出しされても1件しか保存されない。"""
    store = MemoryStore(str(tmp_path / "memory.json"))
    for _ in range(4):
        store.remember("好きな色は青緑色(ティール)です")
    assert store._load() == ["好きな色は青緑色(ティール)です"]


def test_recall_filters_by_query(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.remember("好きな色は青緑色")
    store.remember("好きな食べ物は寿司")
    assert "青緑色" in store.recall("色")
    assert "寿司" not in store.recall("色")
    assert "該当する記憶はありません" in store.recall("車")
