import io

from local_automata_core.colors import BOLD_CYAN, RESET, colorize


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_colorize_on_tty():
    out = colorize("hi", BOLD_CYAN, stream=_Tty())
    assert out == f"{BOLD_CYAN}hi{RESET}"


def test_colorize_plain_when_not_tty():
    assert colorize("hi", BOLD_CYAN, stream=io.StringIO()) == "hi"


def test_no_color_env_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert colorize("hi", BOLD_CYAN, stream=_Tty()) == "hi"
