import os.path

from lark import Lark
from .exp_parser import ExpTree
from .language import Language, DataType
from .logic import Predicate, NeuralPredicate, FuncSymbol, Const


class DataUtils(object):
    """Utilities about logic.

    A class of utilities about first-order logic.

    Args:
        dataset_type (str): A dataset type (kandinsky or clevr).
        dataset (str): A dataset to be used.

    Attrs:
        base_path: The base path of the dataset.
    """

    def __init__(self, lark_path, lang_base_path, dataset_type='kandinsky', dataset='twopairs'):
        self.base_path = lang_base_path + dataset_type + '/' + dataset + '/'
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    def load_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        with open(path) as f:
            for line in f:
                tree = self.lp_clause.parse(line[:-1])
                clause = ExpTree(lang).transform(tree)
                clauses.append(clause)
        return clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    tree = self.lp_atom.parse(line[:-2])
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    def load_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_pred(line) for line in lines]
        return preds

    def load_neural_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_neural_pred(line) for line in lines]
        return preds

    def load_consts(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            consts.extend(self.parse_const(line))
        return consts

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return Predicate(pred, int(arity), dtypes)

    def parse_neural_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return NeuralPredicate(pred, int(arity), dtypes)

    def parse_funcs(self, line):
        """Parse string to function symbols.
        """
        funcs = []
        for func_arity in line.split(','):
            func, arity = func_arity.split(':')
            funcs.append(FuncSymbol(func, int(arity)))
        return funcs

    def parse_const(self, line):
        """Parse string to function symbols.
        """
        line = line.replace('\n', '')
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(',')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_clauses(self, lang):
        return self.load_clauses(self.base_path + 'clauses.txt', lang)

    def get_bk(self, lang):
        return self.load_atoms(self.base_path + 'bk.txt', lang)

    def load_language(self):
        """Load language, background knowledge, and clauses from files.
        """
        preds = self.load_preds(self.base_path + 'preds.txt') + \
            self.load_neural_preds(self.base_path + 'neural_preds.txt')
        consts = self.load_consts(self.base_path + 'consts.txt')
        lang = Language(preds, [], consts)
        return lang
