import torch
from torch import nn

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset, MoleculeNet

from pooling.attn_pooling import GraphSelfAttention

from tqdm import tqdm

class meanGCN(nn.Module):
    def __init__(self,  num_features, device='cpu'):
        super().__init__()
        self.device = device

        self.conv1 = GCNConv(num_features, 128)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.conv2 = GCNConv(128, 128)
        self.agg = global_mean_pool

        self.head1 = nn.Linear(128, 64)
        self.head2 = nn.Linear(64, 1)

        self.act2 = nn.Sigmoid()
        
        self.to(device)

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch.to(self.device)

        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.act(x)

        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.act(x)

        x = self.dropout(x)
        
        x = self.agg(x, batch)
        
        x = self.head1(x)
        x = self.act(x)
        x = self.head2(x)
        return x# self.act2(x)

class attnGCN(nn.Module):
    def __init__(self,  num_features, device='cpu'):
        super().__init__()
        self.device = device

        self.conv1 = GCNConv(num_features, 128)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.conv2 = GCNConv(128, 128)

        self.readout = GraphSelfAttention(128, 128, mode='mean')
        # self.agg = global_attn_pool_mean

        self.head1 = nn.Linear(128, 64)
        self.head2 = nn.Linear(64, 1)

        self.act2 = nn.Sigmoid()
        
        self.to(device)

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch.to(self.device)

        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.act(x)

        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.act(x)

        x = self.dropout(x)
        
        x = self.readout(x, batch)
        
        x = self.head1(x)
        x = self.act(x)
        x = self.head2(x)
        return x# self.act2(x)

def train_model(model, train_loader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in (pbar := tqdm(range(n_epochs))):
        total_loss = 0
        for train_batch in train_loader:
            optimizer.zero_grad()

            out = model(train_batch)
            loss = loss_fn(out, train_batch.y.to(model.device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        pbar.set_description(f'Loss: {total_loss/len(train_loader):.4f}')

def load_dataset():
    dataset = MoleculeNet(root='./datasets/BACE', name='BACE').shuffle()

    train_set = dataset[:int(len(dataset)*0.8)]
    test_set = dataset[int(len(dataset)*0.8):]

    train_loader = DataLoader(train_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    return dataset, train_loader, test_loader

def eval_model(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    loss_fn = nn.BCEWithLogitsLoss()

    # Disable gradient calculation for efficiency and to prevent leakage
    with torch.no_grad():
        for data in test_loader:
            
            # Forward pass: 
            # Note: For Graph Neural Networks, we typically pass atom features (x), 
            # connectivity (edge_index), and the graph identifier (batch).
            out = model(data)
            
            # Calculate loss
            loss = loss_fn(out, data.y.to(model.device))
            total_loss += loss.item() * data.num_graphs
            
            # Calculate accuracy
            pred = (out > 0).to(int).to(model.device)
            correct += int((pred == data.y.to(model.device)).sum())
            total_samples += data.num_graphs

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


def main():
    print('Loading dataset...')
    dataset,train_loader, test_loader = load_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    meanmodel = meanGCN(dataset.num_features, device=device)
    attnmodel = attnGCN(dataset.num_features, device=device)

    print('Training models...')
    train_model(meanmodel, train_loader, 250)
    train_model(attnmodel, train_loader, 250)

    print('Evaluating models...')
    print('Mean model')
    print(eval_model(meanmodel, test_loader))
    print('Attn model')
    print(eval_model(attnmodel, test_loader))

if __name__ == '__main__':
    main()