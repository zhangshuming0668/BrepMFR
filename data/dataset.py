# -*- coding: utf-8 -*-
import os
import pathlib
from tqdm import tqdm
import random
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from prefetch_generator import BackgroundGenerator

from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CADSynth(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        num_class=25,
    ):  
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.file_paths = []
        self._get_filenames(path, filelist=split+".txt")

    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        with open(str(root_dir / f"{filelist}"), "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(root_dir.rglob(f"*[0-9].bin")):
            if x.stem in file_list:
                self.file_paths.append(x)
        print("Done loading {} files".format(len(self.file_paths)))


    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        if(self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)  # edge_data[num_edges, U_grid, pnt_feature]

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)   # face_type[num_nodes]
        pyg_graph.face_area = graph.ndata["y"].type(torch.float) # face_area[num_nodes]
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)   # face_loop[num_nodes]
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)    # face_loop[num_nodes]
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)  # feature_type[num_nodes]

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)   # edge_type[num_edges]
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)  # edge_len[num_edges]
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)  # edge_ang[num_edges]
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)   # edge_conv[num_edges]

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.node_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]           # edge_input[num_nodes, num_nodes, max_dist]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]        # spatial_pos[num_nodes, num_nodes]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]        # d2_distance[num_nodes, num_nodes, 64]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-1])

        if(torch.max(pyg_graph.label_feature) > 24 or torch.max(pyg_graph.label_feature) < 0):
            print(pyg_graph.data_id)

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        return sample
        
    def _collate(self, batch):  #batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator(
            batch,
            multi_hop_max_dist=16,  # multi_hop_max_dist: max distance of multi-hop edges 大于该值认为这两个节点没有关系，边编码为0
            spatial_pos_max=32,     # spatial_pos_max: max distance of multi-hop edges 大于该值认为这两个节点没有关系，空间编码降为0
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoaderX(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )


class TransferDataset(Dataset):
    def __init__(
            self,
            root_dir_source,
            root_dir_target,
            split="train",
            random_rotate=False,
            num_class=25,
            open_set=0
    ):
        assert split in ("train", "val", "test")
        source_path = pathlib.Path(root_dir_source)
        target_path = pathlib.Path(root_dir_target)
        self.split = split
        self.random_rotate = random_rotate
        self.num_class = num_class
        self.open_set = bool(open_set)

        self.source_file_paths = []
        self.target_file_paths = []
        self._get_filenames(source_path, target_path)


    def _get_filenames(self, source_dir, target_dir):
        if self.split == "train":
            filelist_s = "s_train.txt"
            filelist_t = "t_train.txt"
        elif self.split == "val":
            filelist_s = "s_val.txt"
            filelist_t = "t_val.txt"
        elif self.split == "test":
            filelist_s = "s_test.txt"
            filelist_t = "t_test.txt"

        print(f"Loading source data...")
        with open(str(source_dir / f"{filelist_s}"), "r") as f:
            s_file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(source_dir.rglob(f"*[0-9].bin")):
            if x.stem in s_file_list:
                if (self.open_set):
                    bin_file = load_graphs(str(x))
                    face_labels = bin_file[0][0].ndata["f"].type(torch.int)  # [num_nodes]
                    if (torch.max(face_labels) > self.num_class):
                        continue
                self.source_file_paths.append(x)
        print("Done loading {} files".format(len(self.source_file_paths)))

        print(f"Loading target data...")
        with open(str(target_dir / f"{filelist_t}"), "r") as f:
            t_file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(target_dir.rglob(f"*[0-9].bin")):
            if x.stem in t_file_list:
                if (self.open_set):
                    bin_file = load_graphs(str(x))
                    face_labels = bin_file[0][0].ndata["f"].type(torch.int)  # [num_nodes]
                    if (torch.max(face_labels) > self.num_class):
                        continue
                self.target_file_paths.append(x)
        print("Done loading {} files".format(len(self.target_file_paths)))

        if self.split != "test":
            random.shuffle(self.source_file_paths)
            random.shuffle(self.target_file_paths)


    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        if (self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)  # edge_data[num_edges, U_grid, pnt_feature]

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)  # face_type[num_nodes]
        pyg_graph.face_area = graph.ndata["y"].type(torch.float)  # face_area[num_nodes]
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)  # face_loop[num_nodes]
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)   # face_loop[num_nodes]
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)  # feature_type[num_nodes]

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)  # edge_type[num_edges]
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)  # edge_len[num_edges]
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)  # edge_ang[num_edges]
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)  # edge_conv[num_edges]

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]  # edge_input[num_nodes, num_nodes, max_dist]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]  # spatial_pos[num_nodes, num_nodes]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]  # d2_distance[num_nodes, num_nodes, 64]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-1])

        return pyg_graph


    def __len__(self):
        if self.split == "train":
            return max(len(self.source_file_paths), len(self.target_file_paths))
        else:
            return len(self.target_file_paths)

    def __getitem__(self, idx):
        idx_s = idx
        idx_t = idx
        if idx_s >= len(self.source_file_paths):
            idx_s = random.randint(0, len(self.source_file_paths)-1)
        if idx_t >= len(self.target_file_paths):
            idx_t = random.randint(0, len(self.target_file_paths)-1)

        fn_s = self.source_file_paths[idx_s]
        fn_t = self.target_file_paths[idx_t]

        sample_s = self.load_one_graph(fn_s)
        sample_t = self.load_one_graph(fn_t)
        sample = {"source_data": sample_s, "target_data": sample_t}
        return sample

    def _collate(self, batch):  # batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator_st(
            batch,
            multi_hop_max_dist=16,  # multi_hop_max_dist: max distance of multi-hop edges 大于该值认为这两个节点没有关系，边编码为0
            spatial_pos_max=32,  # spatial_pos_max: max distance of multi-hop edges 大于该值认为这两个节点没有关系，空间编码降为0
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoaderX(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )