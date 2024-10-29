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

class Enc_cytoself(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        imsize_compressed=[12, 12, 64],
    ):
        super(Enc_cytoself, self).__init__()

        self.n_latent_dim = n_latent_dim

        self.latent_out = spectral_norm(
            nn.Linear(
                int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_latent_dim, 17),
            nn.Sigmoid()
        )

    def forward(self, x_target):

        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        logit = self.classifier(z)
        return logit

class ProteinDataset(Dataset):
    def __init__(self, csv_file, transform=None, root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform  
        self.root = root
        self.embedding = np.load("embedding_feature/train_vqvec2_balance.npy")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.embedding[idx]
        location = self.data['Location_intensity'][idx]
        location = ast.literal_eval(location)
        label1 = np.ones(17).astype(np.float32)
        label2 = np.zeros(17).astype(np.float32)

        for loc_tuple in location:
            label2[loc_tuple[0]] = 1.0
            if loc_tuple[1] == 2:
                label1[loc_tuple[0]] = 0.5
            elif loc_tuple[1] == 1:
                label1[loc_tuple[0]] = 0.1

        return torch.tensor(embedding), torch.tensor(label1), torch.tensor(label2)
class ProteinDataset_val(Dataset):
    def __init__(self, csv_file, raw_protein, transform=None, root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform  
        self.root = root
        self.raw_protein = pd.read_csv(raw_protein)
        self.embeddings = np.load("embedding_feature/val_vqvec2_balance.npy")
    def __len__(self):
        # return len(self.data)
        return len(self.raw_protein)

    def __getitem__(self, idx):
        protein_idx = self.raw_protein['raw_image_ID'].iloc[idx]
        protein_data2 = self.data[self.data['raw_image_ID'] == protein_idx]
        embeddings = self.embeddings[self.data['raw_image_ID'] == protein_idx]

        protein_data = protein_data2
        location = protein_data['location'].iloc[0]
        location = ast.literal_eval(location)
        label = np.zeros(17).astype(np.float32)
        for i in location:
            label[i] = 1.

        label = torch.tensor(label)

        return embeddings, label

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
    Epochs = 301
    if_load = 1
    model = Enc_cytoself().cuda()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = myAsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, positive_clip=None,
                                 disable_torch_grad_focal_loss=True)
    save_path = '../cytoself_balance/global'
    model_path = os.path.join(save_path, 'model_last.pth')
    csv_path = os.path.join(save_path, 'checkpoints.csv')
    print('if loading model...')
    if if_load == 1:
        if os.path.exists(model_path):
            state = torch.load(model_path,
                               map_location='cpu')
            filtered_dict = state
            model.load_state_dict(filtered_dict,
                                  strict=False)
            print('loaded model over')
            if os.path.exists(os.path.join(save_path, 'saved_states.pkl')):
                with open(os.path.join(save_path, 'saved_states.pkl'), 'rb') as f:
                    loaded_states = pickle.load(f)
                random.setstate(loaded_states['random'])
                np.random.set_state(loaded_states['numpy'])
                torch.set_rng_state(loaded_states['torch'])
                g.set_state(loaded_states['generator'])
                print('loaded last model and random state over')
        else:
            print('no model to load')
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
    else:
        print('Not load model')
    print('done\n')

    train_dataset = ProteinDataset(csv_file='../data_list/opencell_train_data.csv', root='../Data')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=2, generator=g)

    val_dataset = ProteinDataset_val(csv_file='../data_list/opencell_val_data.csv', raw_protein='../data_list/opencell_val_data_protein.csv',
                                     root='../Data')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(max_epoch+1, Epochs+1):
        # train-------------------------------------
        print("starting training, epoch:", epoch)
        train_loss = []
        model.train()
        # for i, (inputData, target) in enumerate(tqdm(itertools.islice(train_dataloader, 2))):
        for inputData, intensity, target in tqdm(train_dataloader):
            inputData = inputData.cuda()
            intensity = intensity.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            output = model(inputData)
            loss = criterion(output, target, intensity)
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # val----------------------------------

        val_loss, aps, apsdict, subset_acc, jaccard_score_micro, jaccard_score_macro, jaccard_score_samples, micro_f1, macro_f1, mcc, hammingloss, matlab_data = validate_multi(
            val_dataloader, model)
        matlab_save_path = os.path.join(save_path, f'output/data-r-6-{epoch}.mat')
        sio.savemat(matlab_save_path, matlab_data)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, aps: {(aps):.5f}'
        print(content)

        res = {
            'checkpoint': 'model-{}.ckpt'.format(epoch),
            'val loss': val_loss,
            'mAP': aps,
            'subset acc': subset_acc,
            # 'blanced acc FALSE': balanced_accuracy_score_False,
            # 'blanced acc TRUE': balanced_accuracy_score_True,
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
            torch.save(model.state_dict(), os.path.join(
                save_path,
                'model-{}.pth'.format(epoch)))  
            torch.save(model.state_dict(), os.path.join(
                save_path, 'model_last.pth'))
            random_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            g_state = g.get_state()
            saved_states = {'random': random_state, 'numpy': np_state, 'torch': torch_state, 'generator': g_state}
            with open(os.path.join(save_path, 'saved_states.pkl'), 'wb') as f:
                pickle.dump(saved_states, f)
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
        for (embeddings, target) in tqdm(loader):
            target = target.cuda()  # (1,17)
            embeddings = embeddings.cuda()  # (1,10,12,12,64)
            embeddings = embeddings.squeeze(0)
            outputs = model(embeddings)
            output_mean = torch.mean(outputs, dim=0)
            logits = torch.unsqueeze(output_mean, dim=0)
            pred = logits.detach()

            PREDS.append(pred)
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
    for i in range(17):
        average_precision[i] = average_precision_score(TARGETS[:, i], PREDS[:, i])
    average_precision["macro"] = average_precision_score(TARGETS, PREDS,
                                                         average="macro")
    print('Average precision:')
    print(average_precision)
    return val_loss, average_precision[
        "macro"], average_precision, subset_acc, jaccard_score_micro, jaccard_score_macro, jaccard_score_samples, micro_f1, macro_f1, mcc, hammingloss, matlab_data

if __name__ == '__main__':
    main()

