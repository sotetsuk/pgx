import torch


def select(pred, on_true, on_false):
    return torch.where(pred, on_true, on_false)


def cond(pred, true_fun, false_fun, *operands):
    return select(pred, true_fun(*operands), false_fun(*operands))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)


def switch(index, branches, *operands):
    index = torch.clip(0, index, len(branches) - 1)
    return branches[index](*operands)


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
