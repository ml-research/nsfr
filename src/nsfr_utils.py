import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import data_clevr
import data_kandinsky
from percept import SlotAttentionPerceptionModule, YOLOPerceptionModule
from facts_converter import FactsConverter
from nsfr import NSFReasoner
from logic_utils import build_infer_module, generate_atoms
from valuation import SlotAttentionValuationModule, YOLOValuationModule
attrs = ['color', 'shape', 'material', 'size']


def valuation_to_attr_string(v, atoms, e, th=0.5):
    """Generate string explanations of the scene.
    """

    st = ''
    for i in range(e):
        st_i = ''
        for j, atom in enumerate(atoms):
            #print(atom, [str(term) for term in atom.terms])
            if 'obj' + str(i) in [str(term) for term in atom.terms] and atom.pred.name in attrs:
                if v[j] > th:
                    prob = np.round(v[j].detach().cpu().numpy(), 2)
                    st_i += str(prob) + ':' + str(atom) + ','
        if st_i != '':
            st_i = st_i[:-1]
            st += st_i + '\n'
    return st


def valuation_to_rel_string(v, atoms, th=0.5):
    l = 100
    st = ''
    n = 0
    for j, atom in enumerate(atoms):
        if v[j] > th and not (atom.pred.name in attrs+['in', '.']):
            prob = np.round(v[j].detach().cpu().numpy(), 2)
            st += str(prob) + ':' + str(atom) + ','
            n += len(str(prob) + ':' + str(atom) + ',')
        if n > l:
            st += '\n'
            n = 0
    return st[:-1] + '\n'


def valuation_to_string(v, atoms, e, th=0.5):
    return valuation_to_attr_string(v, atoms, e, th) + valuation_to_rel_string(v, atoms, th)


def valuations_to_string(V, atoms, e, th=0.5):
    """Generate string explanation of the scenes.
    """
    st = ''
    for i in range(V.size(0)):
        st += 'image ' + str(i) + '\n'
        # for each data in the batch
        st += valuation_to_string(V[i], atoms, e, th)
    return st


def generate_captions(V, atoms, e, th):
    captions = []
    for v in V:
        # for each data in the batch
        captions.append(valuation_to_string(v, atoms, e, th))
    return captions


def save_images_with_captions(imgs, captions, folder, img_id_start, dataset):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if dataset == 'online-pair':
        figsize = (15, 15)
    elif dataset == 'red-triangle':
        figsize = (10, 8)
    else:
        figsize = (12, 6)
    # imgs should be denormalized.
    img_id = img_id_start
    for i, img in enumerate(imgs):
        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(img)
        plt.xlabel(captions[i])
        plt.tight_layout()
        plt.savefig(folder+str(img_id)+'.png')
        img_id += 1
        plt.close()


def denormalize_clevr(imgs):
    """denormalize clevr images
    """
    # normalizing: image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
    return (0.5 * imgs) + 0.5


def denormalize_kandinsky(imgs):
    """denormalize kandinsky images
    """
    return imgs


def to_plot_images_clevr(imgs):
    return [img.permute(1, 2, 0).detach().numpy() for img in denormalize_clevr(imgs)]


def to_plot_images_kandinsky(imgs):
    return [img.permute(1, 2, 0).detach().numpy() for img in denormalize_kandinsky(imgs)]


def get_data_loader(args):
    if args.dataset_type == 'kandinsky':
        return get_kandinsky_loader(args)
    elif args.dataset_type == 'clevr':
        return get_clevr_loader(args)
    else:
        assert 0, 'Invalid dataset type: ' + args.dataset_type


def get_clevr_loader(args):
    dataset_train = data_clevr.CLEVRHans(
        args.dataset, 'train'
    )
    dataset_val = data_clevr.CLEVRHans(
        args.dataset, 'val'
    )
    dataset_test = data_clevr.CLEVRHans(
        args.dataset, 'test'
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def get_kandinsky_loader(args, shuffle=False):
    dataset_train = data_kandinsky.KANDINSKY(
        args.dataset, 'train'
    )
    dataset_val = data_kandinsky.KANDINSKY(
        args.dataset, 'val'
    )
    dataset_test = data_kandinsky.KANDINSKY(
        args.dataset, 'test'
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def get_prob(v_T, NSFR, args):
    if args.dataset_type == 'kandinsky':
        predicted = NSFR.predict(v=v_T, predname='kp')
    elif args.dataset_type == 'clevr':
        if args.dataset == 'clevr-hans3':
            predicted = NSFR.predict_multi(
                v=v_T, prednames=['kp1', 'kp2', 'kp3'])
        if args.dataset == 'clevr-hans7':
            predicted = NSFR.predict_multi(
                v=v_T, prednames=['kp1', 'kp2', 'kp3', 'kp4', 'kp5', 'kp6', 'kp7'])
    return predicted


def get_nsfr_model(args, lang, clauses, atoms, bk, device):
    if args.dataset_type == 'kandinsky':
        PM = YOLOPerceptionModule(e=args.e, d=11, device=device)
        VM = YOLOValuationModule(
            lang=lang, device=device, dataset=args.dataset)
    elif args.dataset_type == 'clevr':
        PM = SlotAttentionPerceptionModule(e=10, d=19, device=device)
        VM = SlotAttentionValuationModule(lang=lang,  device=device)
    else:
        assert False, "Invalid dataset type: " + str(args.dataset_type)
    FC = FactsConverter(lang=lang, perception_module=PM,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=4, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                       infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def get_nsfr_model_train(args, lang, clauses, atoms, bk, device, m):
    if args.dataset_type == 'kandinsky':
        PM = YOLOPerceptionModule(e=args.e, d=11, device=device)
        VM = YOLOValuationModule(
            lang=lang, device=device, dataset=args.dataset)
    elif args.dataset_type == 'clevr':
        PM = SlotAttentionPerceptionModule(e=10, d=19, device=device)
        VM = SlotAttentionValuationModule(lang=lang,  device=device)
    else:
        assert False, "Invalid dataset type: " + str(args.dataset_type)
    FC = FactsConverter(lang=lang, perception_module=PM,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=m, infer_step=4, device=device, train=True)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                       infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def __valuation_to_text(v, atoms, e, th=0.5):
    st = ''
    for i in range(e):
        st_i = 'object ' + str(i) + ': '
        # list of indices of the atoms about obj_i
        atom_indices = []
        atoms = []
        for j, atom in enumerate(atoms):
            terms = atom.terms
            if 'obj' + str(j) in [str(term) for term in terms] and atom.pred.name != 'in':
                if v[j] > th:
                    if len(atom.terms) == 2:
                        st_i += str(atom.terms[1]) + ' '
                    if len(atom.terms) == 1:
                        st_i += str(atom.terms[0])
                indices.append(j)
                atoms.append(atom)
        for j in atom_indices:
            if v[j] > th:
                st_i += ''
        st += st_i + '\n'
    return st[:-2]
