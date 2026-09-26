#!/usr/bin/env python3
"""Compare the production tokenizer with a local DS4.1 tokenizer.json.

Requires tokenizers. Optional frozen request shards add the exact evaluation
prompt text (including control tokens) to the multilingual/code regression set.
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--requests', type=Path, nargs='*', default=[])
    parser.add_argument('--suite', type=Path, help='Also compare rendered high-effort prompts with frozen token requests')
    args = parser.parse_args()
    reference = Tokenizer.from_file(str(args.tokenizer))
    texts = [
        'Hello, world!', 'Write Python: def f(x):\n    return x**2\n',
        '#include <vector>\nstd::vector<int> a = {1,2,3};',
        '1234567890 ١٢٣٤ １２３４', '中文和日本語かなカナ 한글',
        'Привет, свят! Καλημέρα مرحبا', 'é café 👩‍💻 → ∑x²',
        'a\t  b\r\n\n   c  ', "I'm it's we'll aren't x+y=42",
        '<｜begin▁of▁sentence｜><｜User｜>2+2?<｜Assistant｜><think>',
    ]
    randomizer = random.Random(0)
    pieces = ['abc', '12345', '中文', 'かな', '_foo', '-bar', ' +', '\n', '\t',
              'é', '🙂', ' x', 'مرحبا', '  ', '\r\n', 'Привет']
    texts += [''.join(randomizer.choices(pieces, k=25)) for _ in range(100)]
    frozen = {}
    for shard in args.requests:
        for item in json.loads(shard.read_text())['requests']:
            frozen[item['id']] = item
            texts.append(reference.decode(item['token_ids'], skip_special_tokens=False))
    with subprocess.Popen([str(args.harness.resolve()), str(args.gguf)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as process:
        try:
            for i, text in enumerate(texts):
                process.stdin.write(json.dumps({'cmd': 'encode', 'text': text}) + '\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('tokenizer harness exited early')
                got = json.loads(line)
                want = reference.encode(text, add_special_tokens=False).ids
                if got.get('ids') != want:
                    raise AssertionError(f'Case {i}: {text!r}\nExpected {want}\nGot {got}')
            if args.suite:
                for item in json.loads(args.suite.read_text())['items']:
                    expected = frozen[item['id']]
                    process.stdin.write(json.dumps({'cmd':'ds41_render', 'messages':item['messages'],
                                                    'thinking':True, 'reasoning_effort':'high'}) + '\n')
                    process.stdin.flush()
                    rendered = json.loads(process.stdout.readline())
                    if (rendered.get('ids') != expected['token_ids'] or
                        hashlib.sha256(rendered.get('text','').encode()).hexdigest() != expected['rendered_prompt_sha256']):
                        raise AssertionError(f"Rendered prompt differs for {item['id']}")
            process.stdin.write('{"cmd":"quit"}\n')
            process.stdin.flush()
            if process.wait(timeout=10):
                raise RuntimeError('tokenizer harness failed')
        finally:
            if process.poll() is None:
                process.terminate()
    print(f'{len(texts)} DS4.1 tokenizer parity cases passed')


if __name__ == '__main__':
    main()
