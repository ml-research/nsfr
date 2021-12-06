from abc import ABC, abstractmethod

import itertools


def flatten(x): return [z for y in x for z in (
    flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class Term(ABC):
    """Terms in first-order logic.

    An abstract class of terms in first-oder logic.

    Attributes:
        name (str): Name of the term.
        dtype (datatype): Data type of the term.
    """
    @abstractmethod
    def __repr__(self, level=0):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def all_vars(self):
        pass

    @abstractmethod
    def all_consts(self):
        pass

    @abstractmethod
    def all_funcs(self):
        pass

    @abstractmethod
    def max_depth(self):
        pass

    @abstractmethod
    def min_depth(self):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def is_var(self):
        pass


class Const(Term):
    """Constants in first-order logic.

    A class of constants in first-oder logic.

    Attributes:
        name (str): Name of the term.
        dtype (datatype): Data type of the term.
    """

    def __init__(self, name, dtype=None):
        self.name = name
        self.dtype = dtype

    def __repr__(self, level=0):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(other) == Const and self.name == other.name

    def __hash__(self):
        return hash(self.__str__())

    def head(self):
        return self

    def subs(self, target_var, const):
        return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for constant ' + str(self)
        return self

    def all_vars(self):
        return []

    def all_consts(self):
        return [self]

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1

    def is_var(self):
        return 0


class Var(Term):
    """Variables in first-order logic.

    A class of variable in first-oder logic.

    Attributes:
        name (str): Name of the variable.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self, level=0):
        # ret = "\t"*level+repr(self.name)+"\n"
        ret = self.name
        return ret

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(other) == Var and self.name == other.name

    def __hash__(self):
        return hash(self.__str__())

    def head(self):
        return self

    def subs(self, target_var, const):
        if self.name == target_var.name:
            return const
        else:
            return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for constant ' + str(self)
        return self

    def all_vars(self):
        return [self]

    def all_consts(self):
        return []

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1

    def is_var(self):
        return 1


class FuncSymbol():
    """Function symbols in first-order logic.

    A class of function symbols in first-oder logic.

    Attributes:
        name (str): Name of the function.
    """

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name and self.arity == other.arity


class FuncTerm(Term):
    """Term with a function symbol f(t_1, ..., t_n)

    A class of terms that cosist of a function symbol in first-oder logic.

    Attributes:
        func_symbol (FuncSymbol): A function symbolc in the term.
        args (List[Term]): arguments for the function symbol.
    """

    def __init__(self, func_symbol, args):
        assert func_symbol.arity == len(
            args), 'Invalid arguments for function symbol ' + func_symbol.name
        self.func_symbol = func_symbol
        self.args = args

    def __str__(self):
        s = self.func_symbol.name + '('
        for arg in self.args:
            s += arg.__str__() + ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self, level=0):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == FuncTerm:
            if self.func_symbol != other.func_symbol:
                return False
            for i in range(len(self.args)):
                if not self.args[i] == other.args[i]:
                    return False
            return True
        else:
            return False

    def head(self):
        return self.func_symbol

    def pre_order(self, i):
        if i == 0:
            return self.func_symbol
        else:
            return self.pre_order(i-1)

    def get_ith_symbol(self, i):
        return self.to_list()[i]

    def get_ith_term(self, i):
        index = [0]
        result = [Term()]

        def _loop(x, i):
            nonlocal index, result
            if i == index[0]:
                result[0] = x
            else:
                if type(x) == FuncTerm:
                    for term in x.args:
                        index[0] += 1
                        _loop(term, i)
        _loop(self, i)
        return result[0]

    def to_list(self):
        ls = []

        def _to_list(x):
            nonlocal ls
            if type(x) == FuncTerm:
                ls.append(x.func_symbol)
                for term in x.args:
                    _to_list(term)
            else:
                # const or var
                ls.append(x)
        _to_list(self)
        return ls

    def subs(self, target_var, const):
        self.args = [arg.subs(target_var, const) for arg in self.args]
        return self

    def all_vars(self):
        var_list = []
        for arg in self.args:
            var_list += arg.all_vars()
        return var_list

    def all_consts(self):
        const_list = []
        for arg in self.args:
            const_list += arg.all_consts()
        return const_list

    def all_funcs(self):
        func_list = []
        for arg in self.args:
            func_list += arg.all_funcs()
        return [self.func_symbol] + func_list

    def max_depth(self):
        arg_depth = max([arg.max_depth() for arg in self.args])
        return arg_depth+1

    def min_depth(self):
        arg_depth = min([arg.min_depth() for arg in self.args])
        return arg_depth+1

    def size(self):
        size = 1
        for arg in self.args:
            size += arg.size()
        return size

    def is_var(self):
        return 0


class Predicate():
    """Predicats in first-order logic.

    A class of predicates in first-order logic.

    Attributes:
        name (str): A name of the predicate.
        arity (int): The arity of the predicate.
        dtypes (List[DataTypes]): The data types of the arguments for the predicate.
    """

    def __init__(self, name, arity, dtypes):
        self.name = name
        self.arity = arity
        self.dtypes = dtypes  # mode = List[dtype]

    def __str__(self):
        # return self.name
        return self.name + '/' + str(self.arity) + '/' + str(self.dtypes)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == Predicate:
            return self.name == other.name
        else:
            return False


class NeuralPredicate(Predicate):
    """Neural predicats.

    A class of neural predicates, which are associated with a differentiable function.

    Attributes:
        name (str): A name of the predicate.
        arity (int): The arity of the predicate.
        dtypes (List[DataTypes]): The data types of the arguments for the predicate.
    """

    def __init__(self, name, arity, dtypes):
        super(NeuralPredicate, self).__init__(name, arity, dtypes)
        self.name = name
        self.arity = arity
        self.dtypes = dtypes

    def __str__(self):
        return self.name + '/' + str(self.arity) + '/' + str(self.dtypes)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return type(other) == NeuralPredicate and self.name == other.name


class Atom():
    """Atoms in first-oder logic.

    A class of atoms: p(t1, ..., tn)

    Attributes:
        pred (Predicate): A predicate of the atom.
        terms (List[Term]): The terms for the atoms.
    """

    def __init__(self, pred, terms):
        assert pred.arity == len(
            terms), 'Invalid arguments for predicate symbol ' + pred.name
        self.pred = pred
        self.terms = terms
        self.neg_state = False

    def __eq__(self, other):
        if other == None:
            return False
        if self.pred == other.pred:
            for i in range(len(self.terms)):
                if not self.terms[i] == other.terms[i]:
                    return False
            return True
        else:
            return False

    def __str__(self):
        s = self.pred.name + '('
        for arg in self.terms:
            s += arg.__str__() + ','
        s = s[0:-1]
        s += ')'
        return s

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        """comparison < """
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        """comparison > """
        return self.__str__() < other.__str__()

    def subs(self, target_var, const):
        self.terms = [term.subs(target_var, const) for term in self.terms]

    def neg(self):
        self.neg_state = not self.neg_state

    def all_vars(self):
        var_list = []
        for term in self.terms:
            # var_list.append(term.all_vars())
            var_list += term.all_vars()
        return var_list

    def all_consts(self):
        const_list = []
        for term in self.terms:
            const_list += term.all_consts()
        return const_list

    def all_funcs(self):
        func_list = []
        for term in self.terms:
            func_list += term.all_funcs()
        return func_list

    def max_depth(self):
        return max([term.max_depth() for term in self.terms])

    def min_depth(self):
        return min([term.min_depth() for term in self.terms])

    def size(self):
        size = 0
        for term in self.terms:
            size += term.size()
        return size


class Clause():
    """Clauses in first-oder logic.

    A class of clauses in first-order logic: A :- B1, ..., Bn.

    Attributes:
        head (Atom): The head atom.
        body (List[Atom]): The atoms for the body.
    """

    def __init__(self, head, body):
        self.head = head
        self.body = body

    def __str__(self):
        head_str = self.head.__str__()
        body_str = ""
        for bi in self.body:
            body_str += bi.__str__()
            body_str += ','
        body_str = body_str[0:-1]
        body_str += '.'
        return head_str + ':-' + body_str

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        return self.__str__() > other.__str__()

    def is_tautology(self):
        return len(self.body) == 1 and self.body[0] == self.head

    def is_duplicate(self):
        if len(self.body) >= 2:
            es = self.body
            return es == [es[0]] * len(es) if es else False
        return False

    def subs(self, target_var, const):
        if type(self.head) == Atom:
            self.head.subs(target_var, const)
        for bi in self.body:
            bi.subs(target_var, const)

    def all_vars(self):
        var_list = []
        var_list += self.head.all_vars()
        for bi in self.body:
            var_list += bi.all_vars()
        var_list = flatten(var_list)
        # remove duplication
        result = []
        for v in var_list:
            if not v in result:
                result.append(v)
        return result

    def all_consts(self):
        const_list = []
        const_list += self.head.all_consts()
        for bi in self.body:
            const_list += bi.all_consts()
        const_list = flatten(const_list)
        return const_list

    def all_funcs(self):
        func_list = []
        func_list += self.head.all_funcs()
        for bi in self.body:
            func_list += bi.all_funcs()
        func_list = flatten(func_list)
        return func_list

    def max_depth(self):
        depth_list = [self.head.max_depth()]
        for b in self.body:
            depth_list.append(b.max_depth())
        return max(depth_list)

    def min_depth(self):
        depth_list = [self.head.min_depth()]
        for b in self.body:
            depth_list.append(b.min_depth())
        return min(depth_list)

    def size(self):
        size = self.head.size()
        for bi in self.body:
            size += bi.size()
        return size
