"""Dense toy matrices: rows of numbers multiplied in Python (host form), and stored weights multiplied in torch in
FP32 (device form), whatever format they are stored in."""


class Projections:
    def prepare(self, rows):
        return [list(map(float, row)) for row in rows]

    def project(self, weight, activations):
        return [[sum(w * a for w, a in zip(row, x)) for row in weight] for x in activations]

    def dense(self, weight):
        return weight

    def load(self, tensors, format):
        return tensors["weight"].float()

    def matmul(self, x, matrix, out_dtype=None):
        # Row by row: a row's bits never depend on the rows beside it (the slot's row invariance).
        import torch
        y = torch.cat([x[r:r + 1].float() @ matrix.T for r in range(x.shape[0])])
        return y if out_dtype is None else y.to(out_dtype)


def create(context, options, parts):
    return Projections()
