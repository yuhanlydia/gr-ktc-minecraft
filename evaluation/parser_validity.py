from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_FENCE = re.compile(r"```(?:javascript|js)\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    program_code: str
    program_name: str
    exec_code: str


class VoyagerActionParser:
    """Validate generated JavaScript with Voyager's Babel dependency."""

    def __init__(self, mineflayer_directory: str | Path) -> None:
        self.mineflayer_directory = Path(mineflayer_directory).resolve()
        if not (self.mineflayer_directory / "node_modules/@babel/core").is_dir():
            raise FileNotFoundError("Voyager @babel/core dependency is not installed")

    @staticmethod
    def extract_code(response: str) -> str:
        fenced = _FENCE.findall(response)
        return "\n".join(fenced).strip() if fenced else response.strip()

    def parse(self, response: str, timeout_seconds: float = 10.0) -> ParsedAction:
        code = self.extract_code(response)
        script = r"""
const babel = require('@babel/core');
const generator = require('@babel/generator').default;
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  try {
    const payload = JSON.parse(input);
    const ast = babel.parse(payload.code);
    const functions = ast.program.body
      .filter(node => node.type === 'FunctionDeclaration')
      .map(node => ({
        name: node.id && node.id.name,
        async: Boolean(node.async),
        params: node.params.map(param => param.name),
        body: generator(node).code,
      }));
    const main = [...functions].reverse().find(fn => fn.async);
    if (!main) throw new Error('No async function found');
    if (main.params.length !== 1 || main.params[0] !== 'bot') {
      throw new Error(`Main function ${main.name} must take exactly one bot argument`);
    }
    process.stdout.write(JSON.stringify({
      ok: true,
      program_code: functions.map(fn => fn.body).join('\n\n'),
      program_name: main.name,
      exec_code: `await ${main.name}(bot);`,
    }));
  } catch (error) {
    process.stdout.write(JSON.stringify({ok: false, error: String(error.message || error)}));
    process.exitCode = 2;
  }
});
"""
        completed = subprocess.run(
            [
                str(self.mineflayer_directory / "node_modules/.bin/node"),
                "-e",
                script,
            ],
            input=json.dumps({"code": code}),
            text=True,
            cwd=self.mineflayer_directory,
            capture_output=True,
            timeout=timeout_seconds,
        )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Babel parser produced invalid output: {completed.stderr}") from exc
        if not result.get("ok"):
            raise ValueError(result.get("error", "unknown JavaScript parse failure"))
        return ParsedAction(result["program_code"], result["program_name"], result["exec_code"])

    def is_valid(self, response: str) -> bool:
        try:
            self.parse(response)
            return True
        except (ValueError, subprocess.SubprocessError):
            return False
