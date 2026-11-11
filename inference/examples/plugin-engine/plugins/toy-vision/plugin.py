"""A toy model plugin that takes images (crit_plugins.IMAGES): toy-model's step model, whose prepare() puts as many
positions as an image has bytes (up to 3) in place of the image's placeholder token, and those positions read the
image's bytes instead of the placeholder, so what it writes depends on the image."""
from .toy_model.plugin import ToyModel


class ToyVision(ToyModel):
    def prepare(self, ids, images, cancelled):
        placeholder, out, inputs = self.options["image_token"], [], []
        for token in ids:
            if token != placeholder:
                out.append(token)
                continue
            if len(inputs) == len(images):
                raise ValueError("the prompt holds more image placeholders than the request has images")
            pixels = [1 + byte % (self.vocabulary - 1) for byte in images[len(inputs)].data[:3]] or [1]
            out += [placeholder] * len(pixels)
            inputs.append(pixels)
        if len(inputs) != len(images):
            raise ValueError("an image without its placeholder in the prompt")
        return out, inputs

    def open(self, request):
        session = super().open(request)
        runs, prompt, at = iter(request.inputs or ()), list(session.prompt), 0
        while at < len(prompt):
            if prompt[at] == self.options["image_token"] and request.inputs:
                pixels = next(runs)
                prompt[at:at + len(pixels)] = pixels
                at += len(pixels)
            else:
                at += 1
        session.prompt = prompt
        return session


def create(context, options, parts):
    return ToyVision(context, options, parts)
