import itertools
from lark import Lark
from lark import Transformer
from .logic import *


def flatten(x): return [z for y in x for z in (
    flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class ExpTree(Transformer):
    '''Functions to parse strings into logical objects using Lark

    Attrs:
        lang (language): the language of first-order logic.
    '''

    def __init__(self, lang):
        self.lang = lang

    def clause(self, trees):
        head = trees[0]
        body = flatten([trees[1]])
        return Clause(head, body)

    def body(self, trees):
        if len(trees) == 0:
            return []
        elif len(trees) == 1:
            return trees[0]
        else:
            return [trees[0]] + trees[1:]

    def atom(self, trees):
        pred = trees[0]
        args = flatten([trees[1]])
        return Atom(pred, args)

    def args(self, content):
        if len(content) == 1:
            return content[0]
        else:
            return [content[0]] + content[1:]

    def const(self, name):
        dtype = self.lang.get_by_dtype_name(name[0])
        return Const(name[0], dtype)

    def variable(self, name):
        return Var(name[0])

    def functor(self, name):
        func = [f for f in self.lang.funcs if f.name == name[0]][0]
        return func

    def predicate(self, alphas):
        pred = [p for p in self.lang.preds if p.name == alphas[0]][0]
        return pred

    def term(self, content):
        if type(content[0]) == FuncSymbol:
            func = content[0]
            args = flatten([content[1]])
            return FuncTerm(func, args)
        else:
            return content[0]

    def small_chars(self, content):
        return content[0]
