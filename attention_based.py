import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=1):
        super(AttentionMIL, self).__init__()
        
        # 1. Extrator de Features (Pode ser uma CNN se for imagem, aqui uso uma MLP para dados tabulares/séries)
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Mecanismo de Atenção (A Mágica do MIL)
        # Calcula um "peso" para cada instância dentro da bag
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) # Retorna 1 valor de score por instância
        )
        
        # 3. Classificador da Bag
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid() # Assumindo classificação binária (1 ou 0)
        )

    def forward(self, x):
        # x shape esperado: (Batch_size, Num_instancias, Features)
        # Ex: (1, 50, 20) -> 1 bag, 50 instâncias, 20 features cada
        
        B, N, F = x.size()
        
        # Achata para passar na rede: (B*N, F)
        x_flat = x.view(-1, F)
        
        # Extrai as features individuais: h shape (B*N, hidden_dim)
        h = self.feature_extractor(x_flat)
        
        # Calcula os scores de atenção não normalizados: a shape (B*N, 1)
        a = self.attention(h)
        
        # Volta pro formato de Bag: (B, N, 1)
        a = a.view(B, N, 1)
        h = h.view(B, N, -1)
        
        # Aplica Softmax para que os pesos de atenção de cada bag somem 1
        A = F.softmax(a, dim=1) 
        
        # Pooling Ponderado (Multiplica as features pelo peso e soma)
        # A.transpose(1, 2) fica (B, 1, N)
        # h é (B, N, hidden_dim)
        # bmm (Batch Matrix Multiply) resulta em (B, 1, hidden_dim)
        z = torch.bmm(A.transpose(1, 2), h) 
        
        # Remove a dimensão extra: (B, hidden_dim)
        z = z.squeeze(1)
        
        # Classifica a bag inteira
        Y_prob = self.classifier(z)
        
        # Retornamos também o 'A' (pesos de atenção) para interpretabilidade!
        # Assim você pode ver QUAIS instâncias o modelo achou que eram o ruído.
        return Y_prob, A