import argparse

import numpy as np
from sklearn.metrics import accuracy_score, recall_score
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nsfr_utils import denormalize_kandinsky, get_data_loader, get_prob, get_nsfr_model
from nsfr_utils import save_images_with_captions, to_plot_images_kandinsky, generate_captions
from logic_utils import get_lang


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--e", type=int, default=4,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["twopairs", "threepairs", "red-triangle", "closeby",
                                              "online", "online-pair", "nine-circles"], help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky",
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
    args = parser.parse_args()
    return args


def predict(NSFR, loader, args, device, writer, th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        predicted = get_prob(V_T, NSFR, args)
        predicted_list.append(predicted)
        target_list.append(target_set)
        if args.plot:
            imgs = to_plot_images_kandinsky(imgs)
            captions = generate_captions(
                V_T, NSFR.atoms, NSFR.pm.e, th=0.3)
            save_images_with_captions(
                imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size

    predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.cat(target_list, dim=0).to(
        torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set,  [m > thresh for m in predicted], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set,  [m > th for m in predicted], average=None)
        return accuracy, rec_score, th


def main():
    args = get_args()

    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    run_name = 'predict/' + args.dataset
    writer = SummaryWriter(f"runs/{run_name}", purge_step=0)

    # get torch data loader
    train_loader, val_loader,  test_loader = get_data_loader(args)

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)

    # Neuro-Symbolic Forward Reasoner
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, device)

    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(
        NSFR, val_loader, args, device, writer, th=0.33, split='val')

    print("Predicting on training data set...")
    # training split
    acc, rec, th = predict(
        NSFR, train_loader, args, device, writer, th=th_val, split='train')

    print("Predicting on test data set...")
    # test split
    acc_test, rec_test, th_test = predict(
        NSFR, test_loader, args, device, writer, th=th_val, split='test')

    print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


if __name__ == "__main__":
    main()
