"""Compare the native byte matcher with independent stdlib JSON acceptance."""
import json
import random
import subprocess
import sys


def main(executable):
    rng = random.Random(17029)
    strings = ("", "a", 'quote"slash\\', "\n\t\x00", "caf\u00e9", "\U0001f600", "\u2028")

    def value(depth):
        choices = [None, True, False, rng.randint(-100000, 100000), rng.uniform(-1e6, 1e6), rng.choice(strings)]
        if depth:
            choices += [[value(depth - 1) for _ in range(rng.randrange(4))],
                        {rng.choice(strings): value(depth - 1) for _ in range(rng.randrange(4))}]
        return rng.choice(choices)

    corpus = [b"", b"{}", b"[]", b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
              b'{"x":1e999}', b'{"x":"\\uD800"}', b'{"x":"\xed\xa0\x80"}',
              b'{"x":"\xf4\x90\x80\x80"}', b'{"x":00}', b'{"x":1.}', b'{"x":1e+}',
              b'{"x":[0,]}', b'{"x":true,}', b'{}\x0b', b'{"x":"\x00"}']
    alphabet = b'{}[],:"\\nulltruefalse0123456789.-+eE \n\t\r\x00\xc3\xa9'
    for _ in range(240):
        document = {rng.choice(strings): value(3) for _ in range(rng.randrange(5))}
        for ascii_only in (True, False):
            raw = json.dumps(document, ensure_ascii=ascii_only, separators=(",", ":")).encode("utf-8")
            corpus.append(raw)
            corpus.extend(raw[:i] for i in range(len(raw)))
            for _ in range(3):
                at = rng.randrange(len(raw))
                corpus.append(raw[:at] + bytes([rng.choice(alphabet)]) + raw[at + 1:])
    corpus.extend(bytes(rng.choice(alphabet) for _ in range(rng.randrange(80))) for _ in range(500))
    corpus = list(dict.fromkeys(corpus))

    def reject_constant(_):
        raise ValueError("non-JSON constant")

    def accepted(raw):
        try:
            return isinstance(json.loads(raw.decode("utf-8"), parse_constant=reject_constant), dict)
        except (UnicodeError, ValueError):
            return False

    completed = subprocess.run([executable, "json_corpus"], input="\n".join(raw.hex() for raw in corpus) + "\n",
                               text=True, encoding="ascii", capture_output=True, check=True, timeout=60)
    answers = completed.stdout.splitlines()
    if len(answers) != len(corpus):
        raise AssertionError(f"Expected {len(corpus)} answers, got {len(answers)}: {completed.stderr}")
    for raw, actual in zip(corpus, answers):
        expected = "1" if accepted(raw) else "0"
        if actual != expected:
            raise AssertionError(f"Native={actual}, stdlib={expected} for {raw!r}")
    print(f"Native JSON matches stdlib for {len(corpus)} valid, malformed and truncated byte strings")


if __name__ == "__main__":
    main(sys.argv[1])
