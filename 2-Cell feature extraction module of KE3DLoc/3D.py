from torch import nn
import numpy as np
from utils import spectral_norm
from utils import get_activation
import torch
from torch.utils.data import Dataset
import tifffile
import pandas as pd
import ast
import os
from src.losses import myAsymmetricLoss
import torch.optim
from sklearn.metrics import f1_score, classification_report, average_precision_score, accuracy_score, jaccard_score, balanced_accuracy_score, hamming_loss, top_k_accuracy_score, matthews_corrcoef
import time
from tqdm import tqdm
import random
import scipy.io as sio
import pickle
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
# this DownLayerResidual block from https://github.com/AllenCellModeling/pytorch_integrated_cell
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

class Enc_6block(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        gpu_ids=0,
        n_ch=2,
        # conv_channels_list=[32, 64, 128, 256],
        # imsize_compressed=[8, 8, 4],
        conv_channels_list=[32, 64, 128, 256, 512, 1024],
        imsize_compressed=[2, 2, 1],
        # imsize_compressed=[5, 3, 2],

    ):
        super(Enc_6block, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))

            ch_in = ch_out

        self.down4 = spectral_norm(nn.Conv3d(128, 512, 4, 4, padding=0, bias=True))

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier = nn.Linear(self.n_latent_dim, 17)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_target):

        for target_path in self.target_path:
            x_target = target_path(x_target)

        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        logit = self.classifier(z)
        logits = self.sigmoid(logit)
        return logits
class Enc_7block(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        gpu_ids=0,
        n_ch=2,
        # conv_channels_list=[32, 64, 128, 256],
        # imsize_compressed=[8, 8, 4],
        conv_channels_list=[32, 64, 128, 256, 512, 1024],
        imsize_compressed=[2, 2, 1],
    ):
        super(Enc_7block, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))

            ch_in = ch_out

        self.transposed_conv = nn.ConvTranspose3d(ch_in, ch_in, 2, 2)
        self.final_down = DownLayerResidual(1024, 2048)
        ch_in = 2048

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier = nn.Linear(self.n_latent_dim, 17)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_target):
        for target_path in self.target_path:
            x_target = target_path(x_target)

        x_target = self.transposed_conv(x_target)
        x_target = self.final_down(x_target)

        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        logit = self.classifier(z)
        logits = self.sigmoid(logit)
        return logits
class Enc_8block(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        gpu_ids=0,
        n_ch=2,
        # conv_channels_list=[32, 64, 128, 256],
        # imsize_compressed=[8, 8, 4],
        conv_channels_list=[32, 64, 128, 256, 512, 1024],
        imsize_compressed=[2, 2, 1],
        # imsize_compressed=[5, 3, 2],

    ):
        super(Enc_8block, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))

            ch_in = ch_out

        self.final_down = nn.Sequential(DownLayerResidual(1024, 2048), DownLayerResidual(2048, 4096))
        ch_in = 4096

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier = nn.Linear(self.n_latent_dim, 17)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_target):
        for target_path in self.target_path:
            x_target = target_path(x_target)
        x_target = PadLayer(pad_dims=(2 + 4, 2 + 4, 1 + 2))(x_target)

        x_target = self.final_down(x_target)

        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        logit = self.classifier(z)
        logits = self.sigmoid(logit)
        return logits

class ProteinDataset(Dataset):
    def __init__(self, csv_file, transform=None, root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data['path'][idx]
        location = self.data['Location_intensity'][idx]
        location = ast.literal_eval(location)
        image = tifffile.imread(os.path.join(self.root, image_path)).astype(np.float32)
        image = image.transpose([0, 2, 3, 1])  # (2, 64, 128, 128) to (2, 128, 128, 64)
        if self.transform is not None:
            image = self.transform(image)

        label1 = np.ones(17).astype(np.float32)  # intensity
        label2 = np.zeros(17).astype(np.float32)  # label
        for loc_tuple in location:
            label2[loc_tuple[0]] = 1.0  # label to 1
            # if loc_tuple[1] == 3:  # Intensity has defaulted to 1
            #     label1[loc_tuple[0]] = 1.0
            if loc_tuple[1] == 2:
                label1[loc_tuple[0]] = 0.5
            elif loc_tuple[1] == 1:
                label1[loc_tuple[0]] = 0.1
        return torch.tensor(image), torch.tensor(label1), torch.tensor(label2)

class ProteinDataset_val(Dataset):
    def __init__(self, csv_file, raw_protein, transform=None, root=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root = root
        self.raw_protein = pd.read_csv(raw_protein)

    def __len__(self):
        # return len(self.data)
        return len(self.raw_protein)

    def __getitem__(self, idx):
        protein_idx = self.raw_protein['raw_image_ID'].iloc[idx]  # Row corresponding to the protein
        protein_data2 = self.data[self.data['raw_image_ID'] == protein_idx]  # images corresponding to the protein

        protein_data = protein_data2
        location = protein_data['location'].iloc[0]  # label
        location = ast.literal_eval(location)
        label = np.zeros(17).astype(np.float32)
        for i in location:
            label[i] = 1.

        label = torch.tensor(label)

        examples = []
        for idx in protein_data.index:  # image list
            image_path = os.path.join(self.root, protein_data['path'][idx])
            examples.append(image_path)

        return examples, label

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
    model = Enc_6block().cuda()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = myAsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, positive_clip=None, disable_torch_grad_focal_loss=True)  
    save_path = '../model_6block_segblock_12_loss/1-raw'
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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=2,
                                                   generator=g)

    val_dataset = ProteinDataset_val(csv_file='../data_list/opencell_val_data.csv', raw_protein='../data_list/opencell_val_data_protein.csv',
                                     root='../Data')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(max_epoch+1, Epochs+1):
        # train-------------------------------------
        print("starting training, epoch:", epoch)
        train_loss = []
        model.train()
        # for i, (inputData, target) in enumerate(tqdm(itertools.islice(train_dataloader, 20))):
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
        for (datapaths, target) in tqdm(loader):
            target = target.cuda()
            batch_size = 6
            outputs = []
            for i in range(0, len(datapaths), batch_size):
                examples = []
                batch = datapaths[i:i + batch_size]

                for path in batch:
                    image = tifffile.imread(path).astype(np.float32)
                    image = image.transpose([0, 2, 3, 1])  # (2, 64, 128, 128) to (2, 128, 128, 64)
                    examples.append(torch.tensor(image))

                images_batch = torch.stack(examples).cuda()
                output = model(images_batch)
                outputs.append(output)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

            outputs = torch.cat(outputs, dim=0)
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
