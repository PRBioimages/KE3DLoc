from utils import spectral_norm
from utils import get_activation
from src.losses import myAsymmetricLoss, ASLSingleLabel
from src.KGEmodel import KGEModel

import numpy as np
import tifffile
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim
from sklearn.metrics import f1_score, classification_report, average_precision_score, accuracy_score, jaccard_score, \
    hamming_loss, matthews_corrcoef
import time
import random
import os
from tqdm import tqdm
import pickle
import scipy.io as sio
import re

class PadLayer(nn.Module):
    def __init__(self, pad_dims):
        super(PadLayer, self).__init__()

        self.pad_dims = pad_dims

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(
                x,
                [0, self.pad_dims[2], 0, self.pad_dims[1], 0, self.pad_dims[0]],
                "constant",
                0,
            )

class DownLayerResidual(nn.Module):
    def __init__(self, ch_in, ch_out, activation="relu", activation_last=None):
        super(DownLayerResidual, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=0),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )
        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm3d(ch_out),
        )

        self.activation = get_activation(activation_last)

    def forward(self, x, x_proj=None, x_class=None):

        x = self.bypass(x) + self.resid(x)

        x = self.activation(x)

        return x

class Enc_5block(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        gpu_ids=0,
        n_ch=2,
        # conv_channels_list=[32, 64, 128, 256],
        # imsize_compressed=[8, 8, 4],
        conv_channels_list=[32, 64, 128, 256, 512],
        imsize_compressed=[4, 4, 2],
    ):
        super(Enc_5block, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))

            ch_in = ch_out

        self.down4 = spectral_norm(nn.Conv3d(128, 512, 4, 4, padding=0, bias=True))
        ch_in = 512

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier19 = nn.Linear(self.n_latent_dim, 19)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
    def freeze_encoder(self, if_freeze=False):
        if if_freeze:
            for param in self.target_path.parameters():
                param.requires_grad = False
            # for param in self.latent_out.parameters():
            #     param.requires_grad = False
        else:
            pass
    def forward(self, x_target):

        for target_path in self.target_path:
            x_target = target_path(x_target)

        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        logit = self.classifier19(z)
        logits = self.LogSoftmax(logit)
        return logits

class ProteinDataset(Dataset):
    def __init__(self, csv_file, transform=None, root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data['save_reg_path'][idx]
        location = self.data['Location'][idx]
        image = tifffile.imread(os.path.join(self.root, image_path)).astype(np.float32)
        image = image.transpose([0, 2, 3, 1])
        for i in range(image.shape[0]):
            channel = image[i]
            max_value = np.max(channel)
            min_value = np.min(channel)
            image[i] = (channel - min_value) / (max_value - min_value)

        if np.isnan(image).any():
            print(f"NaN found in image: {image_path}, replacing NaNs with 0.")
            image = np.nan_to_num(image, nan=0.0)

        label2 = np.zeros(19).astype(np.float32)
        label2[location] = 1.0
        return torch.tensor(image), torch.tensor(location), torch.tensor(label2)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # “You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):”——pytorch
    if torch.cuda.is_available():
        print('torch.cuda.is_available()', torch.cuda.is_available())
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ^^ safe to call this function even if cuda is not available
    # if is_tf_available():
    #     tf.random.set_seed(seed)

def main():
    seed = 42
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    max_epoch = 0
    highest_mAP = 0
    Epochs = 30
    if_load = 1
    model_3D = Enc_5block().cuda()
    model = model_3D
    model.freeze_encoder(if_freeze=True)
    model.train()
    # model.check_freeze_status()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = ASLSingleLabel(gamma_neg=4, gamma_pos=0, eps=0.1)
    save_path = 'pretrain/Allen_Cell'
    raw_model_path = 'pretrain/'
    model_path = os.path.join(save_path, 'model_last.pth')
    csv_path = os.path.join(save_path, 'checkpoints.csv')
    print('if loading model...')
    if if_load == 1:
        if os.path.exists(csv_path):
            checkpoints = pd.read_csv(csv_path)
            checkpoints.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            highest_mAP = np.max(checkpoints['mAP'])
            print('loaded checkpoints.csv over, highest_mAP:', highest_mAP)
            model_name = checkpoints['checkpoint']
            epoch_numbers = [int(re.search(r'\d+', name).group()) for name in model_name]
            max_epoch = max(epoch_numbers)
            print('max_epoch', max_epoch)
        else:
            checkpoints = pd.DataFrame()
            print('bulid a new csv, checkpoints')
        if max_epoch == 0:
            state_3D = torch.load(os.path.join(raw_model_path, 'StatisticalCell-encoder.pth'))
            model.load_state_dict(state_3D['model'], strict=False)
            print('load 3D model over')
        elif os.path.exists(model_path):
            model_state = torch.load(model_path,
                                     map_location='cpu')
            model.load_state_dict(model_state['model_state_dict'],
                                  strict=False)
            random.setstate(model_state['random_state'])
            np.random.set_state(model_state['numpy_state'])
            torch.set_rng_state(model_state['torch_state'])
            g.set_state(model_state['generator_state'])
            print("Model and optimizer loaded successfully!")
        else:
            print('no model to load')
    else:
        print('Not load model')
    print('done\n')

    train_dataset = ProteinDataset(csv_file='../data_list/Allen_Cell_train_data.csv', root='../Data')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=2,
                                               generator=g)

    val_dataset = ProteinDataset(csv_file='../data_list/Allen_Cell_val_data.csv', root='../Data')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False)

    for epoch in range(max_epoch+1, Epochs+1):
        # train-------------------------------------
        print("starting training, epoch:", epoch)
        train_loss = []
        model.train()
        # model.check_freeze_status()
        # for i, (inputData, target, _) in enumerate(tqdm(itertools.islice(train_dataloader, 3))):
        for (inputData, target, _) in tqdm(train_dataloader):
            inputData = inputData.cuda()  
            target = target.cuda()
            optimizer.zero_grad()
            output = model(inputData)
            loss = criterion(output, target)
            loss.backward()
            train_loss.append(loss.detach().cpu().numpy())
            optimizer.step()

        train_loss = np.mean(train_loss)

        # val----------------------------------

        val_loss, aps, apsdict, subset_acc, jaccard_score_micro, jaccard_score_macro, jaccard_score_samples, micro_f1, macro_f1, mcc, hammingloss, matlab_data = validate_multi(
            val_dataloader, model)

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, aps: {(aps):.5f}'
        print(content)

        res = {
            'checkpoint': 'model-{}.ckpt'.format(epoch),
            'location loss': train_loss,
            'val loss': val_loss,
            'mAP': aps,
            'subset acc': subset_acc,
            'micro F1': micro_f1,
            'macro F1': macro_f1,
            'jaccard score micro': jaccard_score_micro,
            'jaccard score macro': jaccard_score_macro,
            'jaccard score samples': jaccard_score_samples,
            'Matthews Correlation Coefficient': mcc,
            'haming loss': hammingloss,
            'aps_dict': apsdict,
        }

        res_str = f'''
                                            Model: {res["checkpoint"]}
                                            Test Loss: {res["val loss"]}
                                            mAP: {res["mAP"]}
                                            mAP Dictionary: {res["aps_dict"]}
                                            Subset Accuracy: {res["subset acc"]}
                                            Micro F1: {res["micro F1"]}
                                            Macro F1: {res["macro F1"]}
                                            Jaccard Score Micro: {res["jaccard score micro"]}
                                            Jaccard Score Macro: {res["jaccard score macro"]}
                                            Jaccard Score Sample: {res["jaccard score samples"]}
                                            Matthews Correlation Coefficient: {res["Matthews Correlation Coefficient"]}
                                            Hamming Loss: {res["haming loss"]}
                                            '''
        print(res_str)

        res_df = pd.DataFrame([res])
        checkpoints = pd.concat([checkpoints, res_df], ignore_index=True)
        checkpoints.to_csv(csv_path, index=False)

        try:
            model_state = {
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state(),
                'torch_state': torch.get_rng_state(),
                'generator_state': g.get_state()
            }
            torch.save(model_state, os.path.join(save_path, 'model-{}.pth'.format(epoch)))  
            torch.save(model_state, os.path.join(save_path, 'model_last.pth'))
        except Exception as e:
            print("Error occurred while saving model: ", e)
            pass

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

def validate_multi(loader, model):
    model.eval()
    print("starting validation")
    device = torch.device('cuda')

    PREDS = []
    TARGETS = []
    val_loss = []

    with torch.no_grad():
        # for i, (data, _, target) in enumerate(itertools.islice(loader, 2)):
        for (data, _, target) in tqdm(loader):
            data, target = data.cuda(), target.cuda()
            logits = model(data)
            if torch.isnan(logits).any():
                print("NaN found in logits during validation, replacing NaNs with 0.")
                print(logits)
                logits = torch.nan_to_num(logits, nan=0.0)
            pred = logits.detach()
            l = torch.exp(pred)
            PREDS.append(l)
            TARGETS.append(target)

    average_precision = dict()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    matlab_data = {
        'preds': PREDS,
        'targets': TARGETS
    }

    final_probs_Y = (PREDS > 0.5).astype(int)
    print(classification_report(TARGETS, final_probs_Y))
    subset_acc = accuracy_score(TARGETS, final_probs_Y)
    jaccard_score_micro = jaccard_score(TARGETS, final_probs_Y, average='micro')
    jaccard_score_macro = jaccard_score(TARGETS, final_probs_Y, average='macro')
    jaccard_score_samples = jaccard_score(TARGETS, final_probs_Y, average='samples')
    # balanced_accuracy_score_False = balanced_accuracy_score(TARGETS, final_probs_Y, adjusted=False)  #
    # balanced_accuracy_score_True = balanced_accuracy_score(TARGETS, final_probs_Y, adjusted=True)
    micro_f1 = f1_score(TARGETS, final_probs_Y, average='micro')
    macro_f1 = f1_score(TARGETS, final_probs_Y, average='macro')
    mcc = np.mean([matthews_corrcoef(TARGETS[:, i], final_probs_Y[:, i]) for i in range(TARGETS.shape[1])])
    hammingloss = hamming_loss(TARGETS, final_probs_Y)
    for i in range(19):
        average_precision[i] = average_precision_score(TARGETS[:, i], PREDS[:, i])
    average_precision["macro"] = average_precision_score(TARGETS, PREDS,
                                                         average="macro")
    print('Average precision:')
    print(average_precision)
    return val_loss, average_precision[
        "macro"], average_precision, subset_acc, jaccard_score_micro, jaccard_score_macro, jaccard_score_samples, micro_f1, macro_f1, mcc, hammingloss, matlab_data

if __name__ == '__main__':
    main()

