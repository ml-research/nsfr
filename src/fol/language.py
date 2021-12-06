from .logic import Var
import itertools


class Language(object):
    """Language of first-order logic.

    A class of languages in first-order logic.

    Args:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.

    Attrs:
        preds (List[Predicate]): A set of predicate symbols.
        funcs (List[FunctionSymbol]): A set of function symbols.
        consts (List[Const]): A set of constants.
    """

    def __init__(self, preds, funcs, consts):
        self.preds = preds
        self.funcs = funcs
        self.consts = consts

    def __str__(self):
        s = "===Predicates===\n"
        for pred in self.preds:
            s += pred.__str__() + '\n'
        s += "===Function Symbols===\n"
        for func in self.funcs:
            s += func.__str__() + '\n'
        s += "===Constants===\n"
        for const in self.consts:
            s += const.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def get_var_and_dtype(self, atom):
        """Get all variables in an input atom with its dtypes by enumerating variables in the input atom.

        Note:
            with the assumption with function free atoms.

        Args:
            atom (Atom): The atom.

        Returns:
            List of tuples (var, dtype)
        """
        var_dtype_list = []
        for i, arg in enumerate(atom.terms):
            if arg.is_var():
                dtype = atom.pred.dtypes[i]
                var_dtype_list.append((arg, dtype))
        return var_dtype_list

    def get_by_dtype(self, dtype):
        """Get constants that match given dtypes.

        Args:
            dtype (DataType): The data type.

        Returns:
            List of constants whose data type is the given data type.
        """
        return [c for c in self.consts if c.dtype == dtype]

    def get_by_dtype_name(self, dtype_name):
        """Get constants that match given dtype name.

        Args:
            dtype_name (str): The name of the data type to be used.

        Returns:
            List of constants whose datatype has the given name.
        """
        return [c for c in self.consts if c.dtype.name == dtype_name]

    def term_index(self, term):
        """Get the index of a term in the language.

        Args:
            term (Term): The term to be used.

        Returns:
            int: The index of the term.
        """
        terms = self.get_by_dtype(term.dtype)
        return terms.index(term)

    def get_const_by_name(self, const_name):
        """Get the constant by its name.

        Args:
            const_name (str): The name of the constant.

        Returns:
            Const: The matched constant with the given name.

        """
        const = [c for c in self.consts if const_name == c.name]
        assert len(const) == 1, 'Too many match in ' + const_name
        return const[0]

    def get_pred_by_name(self, pred_name):
        """Get the predicate by its name.

        Args:
            pred_name (str): The name of the predicate.

        Returns:
            Predicate: The matched preicate with the given name.
        """
        pred = [pred for pred in self.preds if pred.name == pred_name]
        assert len(pred) == 1, 'Too many or less match in ' + pred_name
        return pred[0]


class DataType(object):
    """Data type in first-order logic.

    A class of data types in first-order logic.

    Args:
        name (str): The name of the data type.

    Attrs:
        name (str): The name of the data type.
    """

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())
