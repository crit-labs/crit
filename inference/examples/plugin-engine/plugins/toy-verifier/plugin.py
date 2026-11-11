"""The toy's verifier: the chain rule."""


class Verifier:
    def accept(self, proposal, predictions, remaining, stops=()):
        tokens = []
        for i in range(min(len(predictions), remaining)):
            tokens.append(predictions[i])
            if predictions[i] in stops or i + 1 >= len(proposal["tokens"]) or proposal["tokens"][i + 1] != predictions[i]:
                break
        return {"tokens": tokens, "accepted": max(0, len(tokens) - 1), "path": list(range(len(tokens)))}


def create(context, options, parts):
    return Verifier()
