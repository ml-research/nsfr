import torch
import torch.nn as nn
from fol.logic import NeuralPredicate
from tqdm import tqdm


class FactsConverter(nn.Module):
    """
    FactsConverter converts the output fromt the perception module to the valuation vector.
    """

    def __init__(self, lang, perception_module, valuation_module, device=None):
        super(FactsConverter, self).__init__()
        self.e = perception_module.e
        self.d = perception_module.d
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z, G, B):
        return self.convert(Z, G, B)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype():
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, B):
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                V[:, i] = self.vm(Z, atom)
            elif atom in B:
                # V[:, i] += 1.0
                V[:, i] += torch.ones((batch_size, )).to(
                    torch.float32).to(self.device)
        V[:, 1] = torch.ones((batch_size, )).to(
            torch.float32).to(self.device)
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == NeuralPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred
