import pytest

from evaluation.parser_validity import VoyagerActionParser


MINEFLAYER = "third_party/voyager/voyager/env/mineflayer"


def test_parser_accepts_complete_voyager_wrapper():
    parser = VoyagerActionParser(MINEFLAYER)
    parsed = parser.parse(
        """```javascript
async function collectLogs(bot) {
  await bot.chat('working');
}
```"""
    )
    assert parsed.program_name == "collectLogs"
    assert parsed.exec_code == "await collectLogs(bot);"


@pytest.mark.parametrize(
    "code",
    [
        "async function broken(bot) {",
        "function syncOnly(bot) { return 1; }",
        "async function wrong(foo) { return foo; }",
    ],
)
def test_parser_rejects_invalid_generate_path(code):
    parser = VoyagerActionParser(MINEFLAYER)
    assert not parser.is_valid(code)

