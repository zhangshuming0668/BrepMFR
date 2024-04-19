# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pathlib
import os

from .modules.brep_encoder import BrepEncoder
from .modules.utils.macro import *
from .modules.domain_adv.domain_discriminator import DomainDiscriminator
from .modules.domain_adv.dann import DomainAdversarialLoss
from .brepseg_model import BrepSeg


class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        x = F.softmax(x, dim=-1)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = self.dense_weight(stacked)
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


class DomainAdapt(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes

        pre_trained_model = BrepSeg.load_from_checkpoint(args.pre_train)
        self.brep_encoder = pre_trained_model.brep_encoder
        self.attention = pre_trained_model.attention
        self.classifier = pre_trained_model.classifier

        # domain adv------------------------------------------------------------------------
        domain_discri = DomainDiscriminator(args.dim_node, hidden_size=512)
        self.domain_adv = DomainAdversarialLoss(domain_discri)
        # domain adv-------------------------------------------------------------------------

        self.pred_s = []
        self.label_s = []
        self.pred_t = []
        self.label_t = []

    def training_step(self, batch, batch_idx):
        self.brep_encoder.train()
        self.attention.train()
        self.classifier.train()
        self.domain_adv.train()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        node_z_s = node_emb_s[node_pos_s]              # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        node_z_t = node_emb_t[node_pos_t]              # [total_nodes, dim]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        z_s = self.attention([node_z_s, graph_z_s])

        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)      # [batch_size]
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        z_t = self.attention([node_z_t, graph_z_t])

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(z_s)  # [total_nodes, num_classes]
        node_seg_t = self.classifier(z_t)  # [total_nodes, num_classes]

        # source classify loss-----------------------------------------------------------------
        num_node_s = node_seg_s.size()[0]
        label_s = batch["label_feature"][:num_node_s].long()
        label_s_onehot = F.one_hot(label_s, self.num_classes)
        loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)
        self.log("train_loss_s", loss_s, on_step=False, on_epoch=True)

        # target classify loss-----------------------------------------------------------------
        num_node_t = node_seg_t.size()[0]
        label_t = batch["label_feature"][num_node_s:].long()  # labels [total_nodes]
        loss_t = EntropyLoss(node_seg_t)
        self.log("train_loss_t", loss_t, on_step=False, on_epoch=True)

        # domain adaptation--------------------------------------------------------------
        max_num_node = max(num_node_s, num_node_t)
        pad_z_s = nn.ZeroPad2d(padding=(0, 0, 0, max_num_node-num_node_s))
        z_s_ = pad_z_s(z_s)
        pad_z_t = nn.ZeroPad2d(padding=(0, 0, 0, max_num_node-num_node_t))
        z_t_ = pad_z_t(z_t)
        weight_s = torch.zeros([max_num_node], device=z_s.device, dtype=z_s.dtype)
        weight_s[:num_node_s] = 1.0
        weight_t = torch.zeros([max_num_node], device=z_t.device, dtype=z_t.dtype)
        weight_t[:num_node_t] = 1.0
        loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        self.log("train_loss_transfer", loss_adv, on_step=False, on_epoch=True)
        self.log("train_transfer_acc", domain_acc, on_step=False, on_epoch=True)

        # ===============================================================================================
        # pre_face_acc
        pred_s = torch.argmax(node_seg_s, dim=-1)  # pres [total_nodes]
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        per_face_comp_s = (pred_s_np == label_s_np).astype(np.int)
        self.log("train_acc_s", np.mean(per_face_comp_s), on_step=True, on_epoch=True)

        pred_t = torch.argmax(node_seg_t, dim=-1)  # pres [total_nodes]
        known_pos = torch.where(label_t < self.num_classes)
        label_t_ = label_t[known_pos]
        pred_t_ = pred_t[known_pos]
        label_t_np = label_t_.long().detach().cpu().numpy()
        pred_t_np = pred_t_.long().detach().cpu().numpy()
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("train_acc_t", np.mean(per_face_comp_t), on_step=True, on_epoch=True)
        # ===============================================================================================

        loss = loss_s + 0.3*loss_adv + 0.1*loss_t
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()
        self.attention.eval()
        self.domain_adv.eval()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        node_z_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        node_z_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(self.device)
        z_s = self.attention([node_z_s, graph_z_s])

        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(self.device)
        z_t = self.attention([node_z_t, graph_z_t])

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(z_s)  # [total_nodes, num_classes]
        node_seg_t = self.classifier(z_t)  # [total_nodes, num_classes]

        # source classify loss----------------------------------------------------------------
        num_node_s = node_seg_s.size()[0]
        label_s = batch["label_feature"][:num_node_s].long()
        label_s_onehot = F.one_hot(label_s, self.num_classes)
        loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)
        self.log("eval_loss_s", loss_s, on_step=False, on_epoch=True)
        # target classify loss----------------------------------------------------------------
        loss_t = EntropyLoss(node_seg_t)
        self.log("eval_loss_t", loss_t, on_step=False, on_epoch=True)

        # pre_face_acc
        pred_s = torch.argmax(node_seg_s, dim=-1)  # pres [total_nodes]
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        for i in range(len(pred_s_np)): self.pred_s.append(pred_s_np[i])
        for i in range(len(label_s_np)): self.label_s.append(label_s_np[i])

        # pre_face_acc-----------------------------------
        label_t = batch["label_feature"][num_node_s:].long()  # labels [total_nodes]
        pred_t = torch.argmax(node_seg_t, dim=-1)  # pres [total_nodes]
        known_pos = torch.where(label_t < self.num_classes)
        label_t_ = label_t[known_pos]
        pred_t_ = pred_t[known_pos]
        label_t_np = label_t_.long().detach().cpu().numpy()
        pred_t_np = pred_t_.long().detach().cpu().numpy()
        for i in range(len(pred_t_np)): self.pred_t.append(pred_t_np[i])
        for i in range(len(label_t_np)): self.label_t.append(label_t_np[i])

        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        loss = 1.0 / np.mean(per_face_comp_t)
        self.log("eval_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        pred_s_np = np.array(self.pred_s)
        label_s_np = np.array(self.label_s)
        self.pred_s = []
        self.label_s = []
        per_face_comp_s = (pred_s_np == label_s_np).astype(np.int)
        self.log("per_face_accuracy_source", np.mean(per_face_comp_s))

        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))

        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))

        print("num_classes: %s" % self.num_classes)
        # pre-class acc-----------------------------------------------------------------------------
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_pred = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp = (class_i_pred == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))


    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()
        self.domain_adv.eval()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        node_z_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        node_z_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        z_s = self.attention([node_z_s, graph_z_s])

        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        z_t = self.attention([node_z_t, graph_z_t])

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(z_s)  # [total_nodes, num_classes]
        node_seg_t = self.classifier(z_t)  # [total_nodes, num_classes]

        pred_t = torch.argmax(F.softmax(node_seg_t, dim=-1), dim=-1)  # pres [total_nodes]
        num_node_s = node_seg_s.size()[0]
        label_t = batch["label_feature"][num_node_s:].long()  # labels [total_nodes]
        known_pos = torch.where(label_t < self.num_classes)
        label_t_ = label_t[known_pos]
        pred_t_ = pred_t[known_pos]
        label_t_np = label_t_.long().detach().cpu().numpy()
        pred_t_np = pred_t_.long().detach().cpu().numpy()
        for i in range(len(pred_t_np)): self.pred_t.append(pred_t_np[i])
        for i in range(len(label_t_np)): self.label_t.append(label_t_np[i])

        # 将结果转为txt文件--------------------------------------------------------------------------
        n_graph, max_n_node = padding_mask_t.size()[:2]
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        face_feature[node_pos_t] = pred_t[:]
        out_face_feature = face_feature.long().detach().cpu().numpy()  # [n_graph, max_n_node]
        for i in range(n_graph):
            # 计算每个graph的实际n_node
            end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int))
            # masked出实际face feature
            pred_feature = out_face_feature[i][:end_index + 1]  # (n_node)

            output_path = pathlib.Path("/home/zhang/datasets_segmentation/2_val")
            file_name = "feature_" + str(batch["id"][n_graph + i].long().detach().cpu().numpy()) + ".txt"
            file_path = os.path.join(output_path, file_name)
            feature_file = open(file_path, mode="a")
            for j in range(end_index):
                feature_file.write(str(pred_feature[j]))
                feature_file.write("\n")
            feature_file.close()

        # Visualization of the features ------------------------------
        # # traget
        # feture_np = node_z_t.detach().cpu().numpy()
        # json_root = {}
        # json_root["node_feature"] = feture_np.tolist()
        # json_root["gt_label"] = label_t_np.tolist()
        # json_root["pred_label"] = pred_t_np.tolist()
        # output_path = pathlib.Path("/home/zhang/datasets_segmentation/3_latent_z/target")
        # file_name = "latent_z_%s.json" % (batch_idx)
        # binfile_path = os.path.join(output_path, file_name)
        # with open(binfile_path, 'w', encoding='utf-8') as fp:
        #     json.dump(json_root, fp, indent=4)
        #
        # # source
        # feture_np = node_z_s.detach().cpu().numpy()
        # json_root = {}
        # json_root["node_feature"] = feture_np.tolist()
        # json_root["gt_label"] = label_t_np.tolist()
        # json_root["pred_label"] = pred_t_np.tolist()
        # output_path = pathlib.Path("/home/zhang/datasets_segmentation/3_latent_z/source")
        # file_name = "latent_z_%s.json" % (batch_idx)
        # binfile_path = os.path.join(output_path, file_name)
        # with open(binfile_path, 'w', encoding='utf-8') as fp:
        #     json.dump(json_root, fp, indent=4)
        # Visualization of the features ------------------------------

    def test_epoch_end(self, outputs):
        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))
        print("num_classes: %s" % self.num_classes)
        print("per_face_accuracy: %s" % np.mean(per_face_comp_t))

        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))
        print("per_face_accuracy_feature: %s" % np.mean(per_face_comp_feature))

        # pre-class acc-----------------------------------------------------------------------------
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        self.log("per_class_accuracy", np.mean(per_class_acc))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))

        # IoU---------------------------------------------------------------------------------------
        per_class_iou = []
        for i in range(0, self.num_classes):
            label_pos = np.where(label_t_np == i)
            pred_pos = np.where(pred_t_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.brep_encoder.parameters(), lr=0.0001, betas=(0.99, 0.999))
        optimizer.add_param_group({'params': self.classifier.parameters(), 'lr': 0.0001, 'betas': (0.99, 0.999)})
        optimizer.add_param_group({'params': self.domain_adv.parameters(), 'lr': 0.001, 'betas': (0.99, 0.999)})

        # 学习策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               threshold=0.0001, threshold_mode='rel',
                                                               min_lr=0.000001, cooldown=2, verbose=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"}
                }
