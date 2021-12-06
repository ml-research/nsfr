import numpy as np
import torch.nn as nn
import torch
from logic_utils import get_index_by_predname


class NSFReasoner(nn.Module):
    """The Neuro-Symbolic Forward Reasoner.

    Args:
        perception_model (nn.Module): The perception model.
        facts_converter (nn.Module): The facts converter module.
        infer_module (nn.Module): The differentiable forward-chaining inference module.
        atoms (list(atom)): The set of ground atoms (facts).
    """

    def __init__(self, perception_module, facts_converter, infer_module, atoms, bk, clauses, train=False):
        super().__init__()
        self.pm = perception_module
        self.fc = facts_converter
        self.im = infer_module
        self.atoms = atoms
        self.bk = bk
        self.clauses = clauses
        self._train = train

    def get_params(self):
        return self.im.get_params()  # + self.fc.get_params()

    def forward(self, x):
        # obtain the object-centric representation
        zs = self.pm(x)
        # convert to the valuation tensor
        V_0 = self.fc(zs, self.atoms, self.bk)
        # perform T-step forward-chaining reasoning
        V_T = self.im(V_0)
        return V_T

    def predict(self, v, predname):
        """Extracting a value from the valuation tensor using a given predicate.
        """
        # v: batch * |atoms|
        target_index = get_index_by_predname(
            pred_str=predname, atoms=self.atoms)
        return v[:, target_index]

    def predict_multi(self, v, prednames):
        """Extracting values from the valuation tensor using given predicates.

        prednames = ['kp1', 'kp2', 'kp3']
        """
        # v: batch * |atoms|
        target_indices = []
        for predname in prednames:
            target_index = get_index_by_predname(
                pred_str=predname, atoms=self.atoms)
            target_indices.append(target_index)
        prob = torch.cat([v[:, i].unsqueeze(-1)
                         for i in target_indices], dim=1)
        B = v.size(0)
        N = len(prednames)
        assert prob.size(0) == B and prob.size(
            1) == N, 'Invalid shape in the prediction.'
        return prob

    def print_program(self):
        """Print asummary of logic programs using continuous weights.
        """
        print('====== LEARNED PROGRAM ======')
        C = self.clauses
        Ws_softmaxed = torch.softmax(self.im.W, 1)

        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_'+str(i)+': ',
                  C[max_i], W_[max_i].detach().cpu().item())

    def print_valuation_batch(self, valuation, n=40):
        self.print_program()
        for b in range(valuation.size(0)):
            print('===== BATCH: ', b, '=====')
            v = valuation[b].detach().cpu().numpy()
            idxs = np.argsort(-v)
            for i in idxs:
                if v[i] > 0.1:
                    print(i, self.atoms[i], ': ', round(v[i], 3))

    def get_valuation_text(self, valuation):
        text_batch = ''  # texts for each batch
        for b in range(valuation.size(0)):
            top_atoms = self.get_top_atoms(valuation[b].detach().cpu().numpy())
            text = '----BATCH ' + str(b) + '----\n'
            text += self.atoms_to_text(top_atoms)
            text += '\n'
            # texts.append(text)
            text_batch += text
        return text_batch

    def get_top_atoms(self, v):
        top_atoms = []
        for i, atom in enumerate(self.atoms):
            if v[i] > 0.7:
                top_atoms.append(atom)
        return top_atoms

    def atoms_to_text(self, atoms):
        text = ''
        for atom in atoms:
            text += str(atom) + ', '
        return text
