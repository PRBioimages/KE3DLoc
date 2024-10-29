from utils import spectral_norm
from utils import get_activation
from src.losses import AsymmetricLoss, myAsymmetricLoss
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        return data

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

        # Right branch of the residual layer
        self.bypass = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=0),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )
        # Trunk of the residual layer
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
class DownLayerResidual_2D(nn.Module):
    def __init__(self, ch_in, ch_out, activation="relu", activation_last=None):
        super(DownLayerResidual_2D, self).__init__()

        if activation_last is None:
            activation_last = activation

        # Right branch of the residual layer
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )
        # Trunk of the residual layer
        self.resid = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
            nn.BatchNorm2d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm2d(ch_out),
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

        self.classifier2 = nn.Sequential(
            nn.Linear(512, 917),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x_target):  # （batch_size,64,w,h,c）
        for target_path in self.target_path:
            x_target = target_path(x_target)  # （batch_size,...,1024）
        x_target = x_target.view(x_target.size()[0], -1)  # (batch_size,...*1024)
        z = self.latent_out(x_target)  
        return z

class Enc_6block_2D(nn.Module):
    def __init__(
        self,
        n_latent_dim=512,
        gpu_ids=0,
        n_ch=2,
        conv_channels_list=[32, 64, 128, 256, 512, 1024],
        imsize_compressed=[2, 2],

    ):
        super(Enc_6block_2D, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual_2D(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual_2D(ch_in, ch_out))

            ch_in = ch_out

        self.down4 = spectral_norm(nn.Conv2d(128, 512, 4, 4, padding=0, bias=True))

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )
        self.classifier = nn.Linear(self.n_latent_dim, 17)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_target):  # （batch_size,64,w,h,c）
        for target_path in self.target_path:
            x_target = target_path(x_target)
        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        return z

class CombinedModel(nn.Module):
    def __init__(self, enc_3d, enc_2d, model_name, nentity, nrelation, hidden_dim, gamma=None, evaluator=None,
                 double_entity_embedding=False,
                 double_relation_embedding=False, triple_relation_embedding=False, quad_relation_embedding=False):
        super(CombinedModel, self).__init__()

        # 3D Encoder
        self.enc_3d = enc_3d

        # 2D Encoder
        self.enc_2d = enc_2d

        # Freezing the layers of the 3D encoder
        for param in self.enc_3d.parameters():
            param.requires_grad = False

        # Freezing the layers of the 2D encoder
        for param in self.enc_2d.parameters():
            param.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Linear(1024, 768),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(768*2, 17),
            nn.Sigmoid()
        )
        # https://github.com/hughxiouge/CompoundE
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        gamma = 12.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim

        if double_relation_embedding:
            self.relation_dim = hidden_dim * 2
        elif triple_relation_embedding:
            self.relation_dim = hidden_dim * 3
        elif quad_relation_embedding:
            self.relation_dim = hidden_dim * 4
        else:
            self.relation_dim = hidden_dim

        self.go_embedding = nn.Embedding(num_embeddings=nentity, embedding_dim=768)
        self.embedding_reduction = nn.Linear(768, 512)

        self.relation_embedding = nn.Embedding(num_embeddings=nrelation, embedding_dim=self.relation_dim)
        embedding_range = 1
        self.relation_embedding.weight.data.uniform_(-self.embedding_range.item(), self.embedding_range.item())

    def forward(self, x_3d=None, head=None, relation=None, tail=None, combine_features=None):
        z, logits, head_embedding, relation_embedding, tail_embedding = None, None, None, None, None

        if head is not None:
            head_embedding = self.go_embedding(head)
            # head_embedding = self.embedding_reduction(head_embedding)
        if relation is not None:
            relation_embedding = self.relation_embedding(relation)
        if tail is not None:
            tail_embedding = self.go_embedding(tail)

        if x_3d is not None:
            x_3d_features = self.enc_3d(x_3d)
            x_2d, _ = torch.max(x_3d, axis=-1)
            x_2d_features = self.enc_2d(x_2d)
            combined_features = torch.cat((x_3d_features, x_2d_features), dim=1)
            z = self.embedding(combined_features)
            # z = z * self.embedding_range

        if combine_features is not None:
            logits = self.classifier(combine_features)

        return z, logits, head_embedding, relation_embedding, tail_embedding

class ProteinDataset(Dataset):
    def __init__(self, csv_file, transform=None, root=None, protein_go_datas=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data['path'][idx]
        location = self.data['Location_intensity'][idx]
        onto_id = self.data['onto_id'][idx]
        location = ast.literal_eval(location)

        image = tifffile.imread(os.path.join(self.root, image_path)).astype(np.float32)
        image = image.transpose([0, 2, 3, 1])  # (2, 64, 128, 128)to(2, 128, 128, 64)
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
        return torch.tensor(image), torch.tensor(label1), torch.tensor(label2), onto_id

class ProteinDataset_val(Dataset):
    def __init__(self, csv_file, raw_protein, transform=None, root=None, protein_go_datas=None):
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
        onto_id = protein_data['onto_id'].iloc[0]

        location = ast.literal_eval(location)
        label = np.zeros(17).astype(np.float32)
        for i in location:
            label[i] = 1.

        label = torch.tensor(label)

        examples = []
        for idx in protein_data.index:  # image list
            image_path = os.path.join(self.root, protein_data['path'][idx])
            examples.append(image_path)

        return examples, label, onto_id

def main():
    seed = 42
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    max_epoch = 0
    highest_mAP = 0
    Epochs = 30
    if_load = 1
    model_3D = Enc_6block().cuda()
    model_2D = Enc_6block_2D().cuda()
    hidden_dim = 384 * 2
    model = CombinedModel(model_3D, model_2D, '', 47229, 65, hidden_dim, double_relation_embedding=True).cuda()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    save_path = '../true_split/new4-protein-go/pairE-1gg-1pg-connect-768+768-lianhe'
    model_path = os.path.join(save_path, 'model_last.pth')

    raw_csv_path = os.path.join(save_path, 'checkpoints.csv')
    csv_path = os.path.join(save_path, 'test-checkpoints.csv')
    print('if loading model...')
    if if_load == 1:
        if os.path.exists(csv_path):
            checkpoints = pd.read_csv(csv_path)
            checkpoints.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            raw_checkpoints = pd.read_csv(raw_csv_path)
            raw_checkpoints.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            highest_mAP = np.max(raw_checkpoints['mAP'])
            print('loaded checkpoints.csv over, highest_mAP:', highest_mAP)
            model_name = raw_checkpoints['checkpoint']
            mcc = raw_checkpoints['Matthews Correlation Coefficient']
            # Find the maximum MCC value and its index
            max_mcc_index = np.argmax(mcc)
            epoch_numbers = [int(re.search(r'\d+', name).group()) for name in model_name]
            max_epoch = epoch_numbers[max_mcc_index]
            print('max_epoch', max_epoch)
        else:
            checkpoints = pd.DataFrame()
            print('bulid a new csv, checkpoints')
            raw_checkpoints = pd.read_csv(raw_csv_path)
            raw_checkpoints.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            highest_mAP = np.max(raw_checkpoints['mAP'])
            print('loaded checkpoints.csv over, highest_mAP:', highest_mAP)
            model_name = raw_checkpoints['checkpoint']
            mcc = raw_checkpoints['Matthews Correlation Coefficient']
            max_mcc_index = np.argmax(mcc)
            epoch_numbers = [int(re.search(r'\d+', name).group()) for name in model_name]
            max_epoch = epoch_numbers[max_mcc_index]
            print('max_epoch', max_epoch)
    else:
        print('Not load model')
    print('done\n')

    with open('best_model/GO_GO_ID_weight.pkl', 'rb') as f:
        go_go_datas = pickle.load(f)
    with open('best_model/Protein_GO_ID_weight.pkl', 'rb') as f:  # relation ID
        protein_go_datas = pickle.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    go_go_datas = to_device(go_go_datas, device)
    protein_go_datas = to_device(protein_go_datas, device)
    go_go_ids = go_go_datas['data']
    protein_go_ids = protein_go_datas['data']
    go_go_weights = go_go_datas['weight']
    protein_go_weights = protein_go_datas['weight']
    criterion = myAsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05,
                                 disable_torch_grad_focal_loss=True)  
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    val_dataset = ProteinDataset_val(csv_file='../../data_list/opencell_test_data.csv', raw_protein='../../data_list/opencell_test_data_protein.csv',
                                     root='../Data', protein_go_datas=protein_go_datas)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    Kmodel = KGEModel(model_name='PairRE', gamma=12.0, hidden_dim=hidden_dim)
    for epoch in ([max_epoch]):
        print("starting training, epoch:", epoch)
        train_loss = []
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        model_path = os.path.join(save_path, f'model-{epoch}.pth')
        state = torch.load(model_path,
                           map_location='cpu')
        filtered_dict = state
        model.load_state_dict(filtered_dict['model_state_dict'],
                              strict=True)

        # val----------------------------------

        val_loss, aps, apsdict, subset_acc, jaccard_score_micro, jaccard_score_macro, jaccard_score_samples, micro_f1, macro_f1, mcc, hammingloss, matlab_data = validate_multi(
            val_dataloader, model, protein_go_ids, Kmodel)
        matlab_save_path = os.path.join(save_path, f'test-{epoch}.mat')
        sio.savemat(matlab_save_path, matlab_data)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, aps: {(aps):.5f}'
        print(content)

        res = {
            'checkpoint': 'model-{}.pth'.format(epoch),
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

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

def validate_multi(loader, model, protein_go_ids, Kmodel):
    model.eval()
    print("starting validation")
    device = torch.device('cuda')

    PREDS = []
    TARGETS = []
    val_loss = []

    with torch.no_grad():
        # for i, (datapaths, target) in enumerate(itertools.islice(loader, 2)):
        # for i, (datapaths, target, onto_id) in enumerate(itertools.islice(loader, 2)):
        for (datapaths, target, onto_id) in tqdm(loader):
            target = target.cuda()
            onto_id = onto_id.cuda()
            batch_size = 6
            outputs = []
            if onto_id != 78962 and onto_id != 214162:
                protein_go_data = protein_go_ids[onto_id.item()]['postive']
                relation = protein_go_data['relation_ids']
                tail_go = protein_go_data['tail_input_ids']
                _, _, _, relation_emb, tail_emb = model(relation=relation, tail=tail_go)
                _, ke_head = Kmodel((torch.zeros(768).cuda(), relation_emb, tail_emb), mode='positive')
                ke_head = torch.mean(ke_head, dim=0)
            else:
                ke_head = torch.zeros(768).cuda()

            for i in range(0, len(datapaths), batch_size):
                examples = []
                batch = datapaths[i:i + batch_size]

                for path in batch:
                    image = tifffile.imread(path).astype(np.float32)
                    image = image.transpose([0, 2, 3, 1])  # (2, 64, 128, 128) to (2, 128, 128, 64)
                    examples.append(torch.tensor(image))

                images_batch = torch.stack(examples).cuda()  # (6,2,128,128,64)
                head_emb, _, _, _, _ = model(x_3d=images_batch)
                H_embedding = ke_head.unsqueeze(0).expand(head_emb.size(0), -1)
                combine_feature = torch.cat((head_emb, H_embedding), dim=-1)
                _, output, _, _, _ = model(combine_features=combine_feature)
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

