import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import torch.nn.parallel
import torch.utils.data
from spatial_utils import *

# normalized layer : to solve the overfit problem
# dropout layer: to solve the overfit problem

class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                    output_dim,
                    dropout_rate=None,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection = False,
                    context_str = ''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN


        "空间关系编码器"（Spatial Relation Encoder）是一种神经网络模块，通常用于处理涉及空间关系的任务，
        特别是在计算机视觉和自然语言处理领域。这个模块的主要目标是捕获输入数据中的空间关系信息，并将其编码成可用于后续任务的表示。

        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 解决过拟合的现象
        # hide serval neurons in the network
        # dropout_rate: the probability of a neuron being dropped out
        # （1）首先随机（临时）删掉网络中一半的隐藏神经元，输入输出神经元保持不变（图3中虚线为部分临时被删除的神经元）
        #
        # （2） 然后把输入x通过修改后的网络前向传播，然后把得到的损失结果通过修改的网络反向传播。一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）。
        #
        # （3）然后继续重复这一过程：
        #
        # 恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）
        # 从隐藏层神经元中随机选择一个一半大小的子集临时删除掉（备份被删除神经元的参数）。
        # 对一小批训练样本，先前向传播然后反向传播损失并根据随机梯度下降法更新参数（w，b） （没有被删除的那一部分参数得到更新，删除的神经元参数保持被删除前的结果）。
        # 不断重复这一过程。
        #
        # present with a probability p
        # if p = 0.5, then half of the neurons will be dropped out


        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        # NLP：
        #
        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same

        # Skip connection
        #  它的基本思想是将网络的某一层的输出直接添加到后续层的输入中，以使梯度能够更顺畅地反向传播，从而有助于解决深度网络中的梯度消失或梯度爆炸问题。

        # 解决梯度消失问题：在深度神经网络中，梯度在反向传播过程中可能会变得非常小，导致网络难以训练。通过跳跃连接，可以将较浅层的梯度直接传递给深层，从而避免梯度消失问题，使梯度更容易传播到较早的层。
        # 反向传播：多层损失的叠加 所以可能会导致梯度消失

        # 梯度爆炸问题： gegenseite von 梯度消失

        # 加速收敛：跳跃连接有助于加速网络的收敛速度，因为它允许网络更快地学习恒等映射（identity mapping），即输出等于输入的情况。
        #
        # 提高网络性能：跳跃连接可以提高网络的性能和泛化能力，因为它使网络能够更好地捕获输入数据中的细微特征和模式。
        #
        # 模型深度扩展：跳跃连接允许构建非常深的神经网络，而不会导致训练困难或性能下降。这对于一些复杂任务和大规模数据集非常有用。
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        #
        # nn.init.xavier_uniform(self.linear.weight) 是PyTorch中的初始化权重参数的操作，具体来说，它使用了一种叫做"Xavier初始化"（也称为Glorot初始化）的方法来初始化线性层（nn.Linear）的权重矩阵 self.linear.weight。
        # Xavier初始化是一种用于神经网络权重初始化的常用方法，旨在有助于加速网络的训练和改善模型的收敛性。这种初始化方法的主要思想是根据网络层的输入和输出的维度来合理地初始化权重，以确保梯度在反向传播时不会过于迅速地消失或爆炸。
        # 具体来说，nn.init.xavier_uniform 的参数 self.linear.weight 表示要初始化的权重矩阵。在 Xavier 初始化中，权重矩阵中的每个元素都会从均匀分布中随机抽样，具体抽样范围的计算如下：
        # 如果权重矩阵的输入维度为 in_features，输出维度为 out_features，则 Xavier 初始化会将权重矩阵中的元素初始化为在区间 [−a,a] 内均匀分布的随机值，其中：
        # 这个初始化方法的思想是使权重的方差在前向传播和反向传播过程中保持差不多的大小，从而帮助训练更加稳定的神经网络。
        # 总之，nn.init.xavier_uniform(self.linear.weight) 是一种初始化权重参数的方法，它通过均匀分布随机初始化权重矩阵的值，并根据输入和输出维度来调整初始化范围，以改善神经网络的训练效果。这种初始化方法在许多深度学习模型中广泛使用。
        nn.init.xavier_uniform(self.linear.weight)
        




    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output

class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                    output_dim,
                    num_hidden_layers=0,
                    dropout_rate=0.5,
                    hidden_dim=-1,
                    activation="relu",
                    use_layernormalize=True,
                    skip_connection = False,
                    context_str = None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()

        # creat layers by using append function of nn.ModuleList()

        if self.num_hidden_layers <= 0:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))
        else:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            for i in range(self.num_hidden_layers-1):
                self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))

        

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output

    """
    freq_init：表示频率列表的初始化方式。可以是 "random" 或 "geometric" 两种选项。
    frequency_num：表示要生成的频率列表中的频率数量。
    max_radius：表示最大半径值。
    min_radius：表示最小半径值。
    函数根据输入的 freq_init 参数的不同，有两种不同的方式来生成频率列表 freq_list：
    
    如果 freq_init 是 "random"，则函数会生成一个随机的频率列表，列表中的每个频率值都在 0 到 max_radius 之间。
    如果 freq_init 是 "geometric"，则函数会生成一个几何级数的频率列表。它首先计算出一组等比数列的频率值，从 min_radius 到 max_radius，然后将这些频率值的倒数作为频率列表中的值。
    最终，函数会返回生成的频率列表 freq_list，该列表包含了根据输入参数计算得到的一组频率值。这个频率列表通常用于后续的空间关系编码过程中，以捕获不同尺度或频率的空间信息。
    """

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num*1.0 - 1))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = 1.0/timescales
    return freq_list


    # 这段代码定义了一个名为 GridCellSpatialRelationEncoder 的PyTorch神经网络模型类，用于编码空间关系信息。以下是该类的主要功能和参数：
    # spa_embed_dim：空间关系嵌入的维度，表示编码后的空间关系信息的维度。
    # coord_dim：空间坐标的维度，通常为2（2D）或3（3D）。
    # frequency_num：不同频率的正弦波的数量，用于编码不同尺度的空间关系。
    # max_radius：模型能够处理的最大上下文半径。
    # min_radius：模型能够处理的最小上下文半径。
    # freq_init：频率列表的初始化方式，可以选择 "random" 或 "geometric"。
    # ffn：可选参数，一个FeedForward神经网络（已在前面的代码中定义），用于对编码后的空间关系信息进行进一步处理。
    # 该类的作用是将给定的坐标信息（通常是一组偏移量 deltaX 和 deltaY）编码成空间关系的嵌入向量。它的主要步骤包括：
    #
    # 根据输入的坐标信息，计算相应的频率列表，这些频率列表将用于正弦波的编码。
    # 根据坐标信息生成输入嵌入（make_input_embeds 函数），该函数会将坐标信息映射到正弦波的空间表示，以编码不同尺度的空间关系。
    # 如果提供了 ffn 参数，将生成的空间关系嵌入向量输入到 FeedForward 神经网络中进行进一步处理。
    # 最终返回编码后的空间关系嵌入向量。
    # 这个类的目的是捕获输入坐标的空间关系信息，以便后续的任务可以利用这些信息进行处理，例如在图神经网络中用于图节点分类或其他任务。

class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
            max_radius =0.01, min_radius = 0.00001,
            freq_init = "geometric",
            ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
          self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")
        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """   
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

class GCN(nn.Module):
    """
        GCN
    """
    def __init__(self, num_features_in=3, num_features_out=1, k=20, MAT=False):
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.MAT = MAT
        # with two convolutional layers and one fully connected layer
        self.conv1 = GCNConv(num_features_in, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)
        if MAT:
          self.fc_morans = nn.Linear(32, num_features_out)
    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
          edge_index = ei
          edge_weight = ew
        else:
          edge_index = knn_graph(c, k=self.k).to(self.device)
          edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        # output of each layer applied activation function
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        if self.MAT:
          morans_output = self.fc_morans(h2)
          return output, morans_output
        else:
          return output

class PEGCN(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """
    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k = 20, MAT=False):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k
        self.MAT = MAT
        # encode for the graph
        self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_hidden_dim,ffn=True,min_radius=1e-06,max_radius=360)
        # Dimensionality reduction
        # from -> to dimension
        # 这种操作通常用于减少特征维度或控制模型复杂度。在神经网络中，通过逐渐减少特征维度，可以在保留关键信息的同时减少计算成本和防止过拟合
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )
        self.conv1 = GCNConv(num_features_in + emb_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)
        if MAT:
          self.fc_morans = nn.Linear(32, num_features_out)
    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
          edge_index = ei
          edge_weight = ew
        else:
          edge_index = knn_graph(c, k=self.k).to(self.device)
          edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        # c.reshape(1, c.shape[0], c.shape[1]) 这一行代码实际上对数组 c 进行了重新形状操作，将其形状改变为 (1, batch_size, num_context_pt)。
        # 这个新的形状具有三个维度，其中第一个维度的大小为1，这种重新形状的操作通常用于确保数据的维度匹配或者是为了满足某些模型的输入要求。

        # 3d matrix
        c = c.reshape(1, c.shape[0], c.shape[1])
        # self.spenc 是一个名为 GridCellSpatialRelationEncoder 的模型（根据你提供的代码），
        # 它用于对坐标信息进行编码。这个编码的目的是将坐标信息转换成一种更有意义或更容易处理的表示形式，通常是为了提取有关空间关系的信息。
        emb = self.spenc(c.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1],emb.shape[2])
        emb = self.dec(emb).float()
        # 这行代码的目的可能是将坐标嵌入（emb）与原始输入特征（x）组合在一起，以便在后续的神经网络层中一起进行处理。这种操作常见于将多个输入源合并成一个输入张量的情况，以供神经网络进行联合学习。
        x = torch.cat((x,emb),dim=1)


        # outputs applied activation function
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        if self.MAT:
          morans_output = self.fc_morans(h2)
          return output, morans_output
        else:
          return output

# 这是一个 PyTorch 模型的损失函数包装器（LossWrapper）类。该类主要用于封装一个模型（model）的前向传播和损失计算过程，以及在多任务学习中处理不同任务的损失。
class LossWrapper(nn.Module):
    def __init__(self, model, task_num=1, loss='mse', uw=True, lamb=0.5, k=20, batch_size=2048):
        super(LossWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.task_num = task_num
        self.uw = uw
        self.lamb = lamb
        self.k = k
        self.batch_size = batch_size
        # if task_num > 1:：这个条件判断语句检查task_num是否大于1，即是否存在多个任务。如果存在多个任务（task_num大于1），则执行下面的代码块。
        # self.log_vars = nn.Parameter(torch.zeros((task_num)))：
        # 在多任务情况下，通常需要为每个任务维护一个损失的权重或方差。这行代码创建了一个名为log_vars的可训练参数（nn.Parameter），
        # 它是一个长度为task_num的向量，初始值都设置为零。这些参数将用于调整不同任务的损失权重或方差。

        # 接下来的部分是设置损失函数，根据传入的loss参数的值选择使用均方误差（MSE）损失函数还是平均绝对误差（L1）损失函数。
        # 如果loss参数的值是"mse"，则创建一个均方误差损失函数（nn.MSELoss()），如果是"l1"，则创建一个平均绝对误差损失函数（nn.L1Loss()）。
        if task_num > 1:
          self.log_vars = nn.Parameter(torch.zeros((task_num)))
        if loss=="mse":
          self.criterion = nn.MSELoss()
        elif loss=="l1":
          self.criterion = nn.L1Loss()

    def forward(self, input, targets, coords, edge_index, edge_weight, morans_input):

        if self.task_num==1:
          outputs = self.model(input, coords, edge_index, edge_weight)
          """
             self.criterion 是在初始化 LossWrapper 类时选择的损失函数，它可以是均方误差（MSE）损失函数或平均绝对误差（L1）损失函数，具体取决于 loss 参数的值。
             targets.float().reshape(-1)：这部分代码首先将 targets 转换为浮点数类型（float），然后使用 .reshape(-1) 将其转换为一维的张量。这是因为通常情况下，
             targets 用于计算损失的目标值是一个二维或多维张量，而模型的输出 outputs 也具有相同的形状。为了计算损失，需要将这两个张量展平成一维，以便逐元素进行比较。
             outputs.float().reshape(-1)：类似地，这部分代码将模型的输出 outputs 转换为浮点数类型（float），然后使用 .reshape(-1) 将其展平成一维张量。
             最后，self.criterion 接受这两个一维张量作为输入，计算它们之间的损失值。这个损失值表示模型的预测与实际目标之间的差异，它是一个衡量模型性能的关键指标。在训练过程中，通过反向传播和优化算法来最小化这个损失，从而不断改善模型的预测能力。"""
          loss = self.criterion(targets.float().reshape(-1),outputs.float().reshape(-1))
          return loss

        else:
          outputs1, outputs2 = self.model(input, coords, edge_index, edge_weight)
          if torch.is_tensor(morans_input):
            targets2 = morans_input
          else:
            moran_weight_matrix = knn_to_adj(knn_graph(coords, k=self.k), self.batch_size) 
            with torch.enable_grad():
              targets2 = lw_tensor_local_moran(targets, sparse.csr_matrix(moran_weight_matrix)).to(self.device)
          if self.uw:
            precision1 = 0.5 * torch.exp(-self.log_vars[0])
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss1 = torch.sum(precision1 * loss1 + self.log_vars[0], -1)

            precision2 = 0.5 * torch.exp(-self.log_vars[1])
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss2 = torch.sum(precision2 * loss2 + self.log_vars[1], -1)

            loss = loss1 + loss2
            loss = torch.mean(loss)
            return loss, self.log_vars.data.tolist()
          else:
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss = loss1 + self.lamb * loss2
            return loss        