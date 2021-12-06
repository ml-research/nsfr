import argparse

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nsfr_utils import denormalize_clevr, get_data_loader, get_prob, get_nsfr_model
from nsfr_utils import save_images_with_captions, to_plot_images_clevr, generate_captions
from logic_utils import get_lang


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--e", type=int, default=10,
                        help="The maximum number of objects in one image")
    parser.add_argument(
        "--dataset", choices=["clevr-hans3", "clevr-hans7"], help="Use clevr-hans dataset.")
    parser.add_argument("--dataset-type", default="clevr",
                        help="kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--plot-cam", action="store_true",
                        help="Plot images cam.")
    args = parser.parse_args()
    return args


def predict(NSFR, loader, args, device, writer, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        #print(valuations_to_string(V_T, NSFR.atoms, NSFR.pm.e))
        predicted = get_prob(V_T, NSFR, args)

        predicted_list.extend(
            list(np.argmax(predicted.detach().cpu().numpy(), axis=1)))
        target_list.extend(
            list(np.argmax(target_set.detach().cpu().numpy(), axis=1)))

        if i < 1:
            if args.dataset_type == 'clevr':
                writer.add_images(
                    'images', denormalize_clevr(imgs).detach().cpu(), 0)
            else:
                writer.add_images(
                    'images', imgs.detach().cpu(), 0)
            writer.add_text('V_T', NSFR.get_valuation_text(V_T), 0)
        if args.plot:
            imgs = to_plot_images_clevr(imgs)
            captions = generate_captions(
                V_T, NSFR.atoms, NSFR.pm.e, th=0.33)
            save_images_with_captions(
                imgs, captions, folder='result/clevr/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size
    predicted = predicted_list
    target = target_list
    return accuracy_score(target, predicted), confusion_matrix(target, predicted)


def main():
    args = get_args()
    assert args.dataset_type == 'clevr', 'Use clever dataset for this script.'
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    run_name = 'predict/' + args.dataset
    writer = SummaryWriter(f"runs/{run_name}", purge_step=0)

    # get torch data loader
    train_loader, val_loader, test_loader = get_data_loader(args)

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)

    # Neuro-Symbolic Forward Reasoner
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, device)
    if len(args.device.split(',')) > 1:
        NSFR = nn.DataParallel(NSFR)

    # validation split
    print("Predicting on validation data set...")
    acc_val, cmat_val = predict(
        NSFR, val_loader, args, device, writer, split='val')

    print("Predicting on training data set...")
    # training split
    acc, cmat = predict(
        NSFR, train_loader, args, device, writer, split='train')

    print("Predicting on test data set...")
    # test split
    acc_test, cmat_test = predict(
        NSFR, test_loader, args, device, writer, split='test')

    print("=== ACCURACY ===")
    print("training acc: ", acc)
    print("val acc: ", acc_val)
    print("test acc: ", acc_test)

    print("=== CONFUSION MATRIX ===")
    print('training:')
    print(cmat)
    print('val:')
    print(cmat_val)
    print('test:')
    print(cmat_test)


if __name__ == "__main__":
    main()
