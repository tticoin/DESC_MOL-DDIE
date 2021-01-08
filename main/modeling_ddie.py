import numpy as np
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, mode, activation):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        if layer_output != 0:
            self.W_output = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                           for _ in range(layer_output)])
            self.W_output_ = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_output)])
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.mode = mode
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        self.activation = activations[activation]

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = self.activation(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fingerprints = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,0,]]
        adjacencies = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,1,]]
        molecular_sizes = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,2,]]
        masks = [torch.tensor(x, dtype=torch.float).to(device) for x in inputs[:,3,]]
        masks = torch.cat(masks).unsqueeze(-1)
        #fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        if self.mode == 'sum':
            molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        elif self.mode == 'mean':
            molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        if self.layer_output != 0:
            for l in self.W_output_:
                molecular_vectors = self.activation(l(molecular_vectors))

        """Mask invalid SMILES vectors"""
        molecular_vectors *= masks

        return molecular_vectors

    def mlp(self, vectors1, vectors2):
        vectors = torch.cat((vectors1, vectors2), 1)
        if self.layer_output != 0:
            for l in self.W_output:
                vectors = torch.relu(l(vectors))
        return vectors


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, args, config, gnn_config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(args.dropout_prob)
    
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        self.activation = activations[args.activation]

        if args.use_cnn:
            self.conv_list = nn.ModuleList([nn.Conv1d(config.hidden_size+2*args.pos_emb_dim, config.hidden_size, w, padding=(w-1)//2) for w in args.conv_window_size])
            self.pos_emb = nn.Embedding(2*args.max_seq_length, args.pos_emb_dim, padding_idx=0)

        if args.use_desc and args.use_mol:
            self.desc_conv = nn.Conv1d(config.hidden_size, args.desc_conv_output_size, args.desc_conv_window_size, padding=(args.desc_conv_window_size-1)//2)
            self.classifier = nn.Linear(config.hidden_size+2*args.desc_conv_output_size+2*gnn_config.dim, config.num_labels)
            self.middle_classifier = nn.Linear(config.hidden_size+2*args.desc_conv_output_size+2*gnn_config.dim, args.middle_layer_size)
        elif args.use_desc:
            self.desc_conv = nn.Conv1d(config.hidden_size, args.desc_conv_output_size, args.desc_conv_window_size, padding=(args.desc_conv_window_size-1)//2)
            if args.desc_layer_hidden != 0: self.W_desc = nn.Linear(2*args.desc_conv_output_size, 2*args.desc_conv_output_size)
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(config.hidden_size+2*args.desc_conv_output_size, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(config.hidden_size+2*args.desc_conv_output_size, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        elif args.use_mol:
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(config.hidden_size+2*gnn_config.dim, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(config.hidden_size+2*gnn_config.dim, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        else:
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        self.init_weights()
        
        if args.use_cnn:
            self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)

        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        if args.use_desc: self.desc_bert = BertModel.from_pretrained(args.model_name_or_path)
        if args.use_mol: self.gnn = MolecularGraphNeuralNetwork(gnn_config.N_fingerprints, gnn_config.dim, gnn_config.layer_hidden, gnn_config.layer_output, gnn_config.mode, gnn_config.activation)

        self.use_cnn = args.use_cnn
        self.use_desc = args.use_desc
        self.desc_layer_hidden = args.desc_layer_hidden
        self.gnn_layer_output = args.gnn_layer_output
        self.use_mol = args.use_mol
        self.middle_layer_size = args.middle_layer_size

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                desc1_ii=None, desc1_am=None, desc1_tti=None,
                desc2_ii=None, desc2_am=None, desc2_tti=None,
                fingerprint=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        #pooled_output = self.dropout(pooled_output)

        if self.use_cnn:
            relative_dist1 *= attention_mask
            relative_dist2 *= attention_mask
            pos_embs1 = self.pos_emb(relative_dist1)
            pos_embs2 = self.pos_emb(relative_dist2)
            conv_input = torch.cat((outputs[0], pos_embs1, pos_embs2), 2)
            conv_outputs = []
            for c in self.conv_list:
                conv_output = self.activation(c(conv_input.transpose(1,2)))
                conv_output, _ = torch.max(conv_output, -1)
                conv_outputs.append(conv_output)
            pooled_output = torch.cat(conv_outputs, 1)

        if self.use_desc:
            desc1_outputs = self.desc_bert(desc1_ii, attention_mask=desc1_am, token_type_ids=desc1_tti)
            desc2_outputs = self.desc_bert(desc2_ii, attention_mask=desc2_am, token_type_ids=desc2_tti)
            desc1_conv_input = desc1_outputs[0]
            desc2_conv_input = desc2_outputs[0]
            desc1_conv_output = self.activation(self.desc_conv(desc1_conv_input.transpose(1,2)))
            desc2_conv_output = self.activation(self.desc_conv(desc2_conv_input.transpose(1,2)))
            pooled_desc1_output, _ = torch.max(desc1_conv_output, -1)
            pooled_desc2_output, _ = torch.max(desc2_conv_output, -1)
            if self.desc_layer_hidden != 0:
                pooled_desc_output = self.activation(self.W_desc(torch.cat((pooled_desc1_output, pooled_desc2_output), 1)))
                pooled_output = torch.cat((pooled_output, pooled_desc_output), 1)
            else:
                pooled_output = torch.cat((pooled_output, pooled_desc1_output, pooled_desc2_output), 1)

        if self.use_mol:
            if fingerprint.ndim == 3: # In case of mini-batchsize = 1
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), 1)
            pooled_output = torch.cat((pooled_output, gnn_output), 1)
        
        pooled_output = self.dropout(pooled_output)
        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt
