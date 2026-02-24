import os
import sys
import logging
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Supondo que essas importações já existam no seu projeto
from data.embedding_service import EmbeddingService
from training_models.repositories.request_repository import RequestsRepository
from training_models.repositories.campaigns_repository import CampaignRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

LABEL_MAP = {"bots": 0, "unsafe": 1}

# ==========================================
# 1. MODELO E DATASET MIL
# ==========================================

class AttentionMIL(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super(AttentionMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        B, N, F_dim = x.size()
        x_flat = x.view(-1, F_dim)
        
        h = self.feature_extractor(x_flat)
        a = self.attention(h)
        
        a = a.view(B, N, 1)
        h = h.view(B, N, -1)
        
        A = F.softmax(a, dim=1) 
        z = torch.bmm(A.transpose(1, 2), h).squeeze(1)
        
        Y_prob = self.classifier(z)
        return Y_prob, A

class MILBagDatasetLogical(Dataset):
    """
    Dataset para MIL que aceita Bags de tamanhos variáveis.
    Cada item do dataset é o histórico completo de um IP.
    """
    def __init__(self, df_bags):
        self.bags = []
        self.bag_labels = []
        self.ips = []
        
        for _, row in df_bags.iterrows():
            # row["embedding"] já é uma lista com N vetores do mesmo IP
            bag_tensor = torch.tensor(row["embedding"], dtype=torch.float32)
            label_tensor = torch.tensor([row["bag_label"]], dtype=torch.float32)
            
            self.bags.append(bag_tensor)
            self.bag_labels.append(label_tensor)
            self.ips.append(row["ip"])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.bag_labels[idx], self.ips[idx]

# ==========================================
# 2. SERVIÇO DE TREINAMENTO MIL
# ==========================================

class MILTrainingService:
    def __init__(
        self, 
        traffic_source: str, 
        repo_requests: RequestsRepository, 
        repo_campaigns: CampaignRepository,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        emb_config: str = "fasttext",
        bag_size: int = 20
    ):
        logger.info("Inicializando MILTrainingService...")
        self.repo_request = repo_requests
        self.repo_campaigns = repo_campaigns
        self.traffic_source = traffic_source
        self.emb_config = emb_config
        self.bag_size = bag_size
        self.limit_cpg: int = 100
        self.limit_each: int = 10000
        self.device = device
        self.model = None
        self.le = LabelEncoder()
        
        self.get_embedding_instance()
        self.emb_dim = EmbeddingService.get_emb_dim()
        logger.info(f"Dimensao do embedding configurada: {self.emb_dim}")

    async def fetch_training_sample(self) -> Tuple[pd.DataFrame, List[Dict]]:
        # ... [MANTIDO IGUAL AO SEU CÓDIGO ORIGINAL] ...
        logger.info(f"Buscando campanhas ativas recentes para a fonte: {self.traffic_source}...")
        recent_campaign_hashes = await self.repo_campaigns.get_recent_active_campaign_hashes(
            traffic_source=self.traffic_source, limit=self.limit_cpg
        )
        if not recent_campaign_hashes: return pd.DataFrame(), {}
        results, hashes_info = await self.repo_request.get_training_sample_by_hashes(
            hashes=recent_campaign_hashes, limit_each=self.limit_each
        )
        return pd.DataFrame(results), hashes_info

    def get_embedding_instance(self):
        # ... [MANTIDO IGUAL AO SEU CÓDIGO ORIGINAL] ...
        model_path = f"G:/Meu Drive/TWR/data/{self.traffic_source}/fasttext_{self.traffic_source}.model" if self.emb_config == "fasttext" else "all-MiniLM-L6-v2"
        EmbeddingService.get_instance(self.emb_config, model_path, traffic_source=self.traffic_source)

    async def generate_embeddings(self) -> Tuple[DataLoader, DataLoader, List[Dict]]:
        logger.info("Iniciando geracao de embeddings e Bags MIL por IP...")
        df, hash_info = await self.fetch_training_sample()

        if df.empty:
            raise ValueError(f"Não há amostras suficientes para a fonte '{self.traffic_source}'.")

        df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
        
        # MAPEAMENTO EXPLÍCITO PARA O MIL
        # Bot (Anomalia) = 1 | Unsafe (Humano/Normal) = 0
        mapeamento_mil = {"bots": 1, "unsafe": 0}
        df["decision_mil"] = df["decision"].map(mapeamento_mil)
        
        embeddings_matrix, texts = EmbeddingService.process_and_encode(df)
        
        # Coloca os embeddings de volta no DataFrame para podermos agrupar
        df["embedding"] = list(embeddings_matrix)

        # ==========================================
        # A MÁGICA DO MIL: AGRUPANDO POR IP
        # ==========================================
        logger.info(f"Agrupando {len(df)} requisições por IPs únicos...")

        bags_df = df.groupby("ip").agg({
            "embedding": list,       # Junta todos os vetores desse IP numa lista
            "decision_mil": list     # Junta todas as labels desse IP numa lista
        }).reset_index()

        # Define a label da Bag: Se o IP fez pelo menos 1 requisição maliciosa (1), a Bag vira 1.
        bags_df["bag_label"] = bags_df["decision_mil"].apply(lambda labels: 1.0 if 1 in labels else 0.0)

        logger.info(f"Criadas {len(bags_df)} Bags lógicas baseadas em IP.")
        print("\nDistribuição Real das Bags (IPs Humanos x IPs de Bots):")
        print(bags_df["bag_label"].value_counts())
        print("-" * 50)

        # ==========================================
        # DIVISÃO TREINO / TESTE (AGORA BASEADA NAS BAGS/IPs)
        # ==========================================
        # Nós cortamos o dataframe 'bags_df', mantendo os históricos dos IPs intactos!
        df_train, df_test = train_test_split(
            bags_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=bags_df["bag_label"] # Garante que a proporção de bots no treino e teste seja a mesma
        )

        print(df_train.head())

        logger.info("Instanciando os Datasets Lógicos...")
        dataset_train = MILBagDatasetLogical(df_train)
        dataset_test = MILBagDatasetLogical(df_test)

        # ATENÇÃO: batch_size=1 é obrigatório! 
        # Um IP pode ter 2 requests (shape 2x300), outro pode ter 50 (shape 50x300).
        # O PyTorch não consegue colocar formatos diferentes no mesmo batch sem complexidade extra.
        train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

        logger.info(f"Gerados {len(dataset_train)} IPs para treino e {len(dataset_test)} IPs para teste.")
        return train_loader, test_loader, hash_info

    async def training_pipeline(self, epochs=10, path="models"):
        train_loader, test_loader, hash_info = await self.generate_embeddings()

        self.model = AttentionMIL(in_features=self.emb_dim, hidden_dim=256).to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss() # Loss para binário

        logger.info("Iniciando treinamento MIL...")
        
        for epoch in range(epochs):
            # Fase de Treino
            self.model.train()
            train_loss, train_preds, train_targets = 0.0, [], []
            
            for bags, labels in train_loader:
                bags, labels = bags.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                preds, _ = self.model(bags)
                loss = criterion(preds, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend((preds > 0.85).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            # Fase de Validação
            self.model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []
            
            with torch.no_grad():
                for bags, labels in test_loader:
                    bags, labels = bags.to(self.device), labels.to(self.device)
                    preds, _ = self.model(bags)
                    loss = criterion(preds, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend((preds > 0.85).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            # Metricas
            t_acc = accuracy_score(train_targets, train_preds)
            v_acc = accuracy_score(val_targets, val_preds)
            t_loss = train_loss / len(train_loader)
            v_loss = val_loss / len(test_loader)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} - Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} - Acc: {v_acc:.4f}")
        
        self.audit_model(model=self.model, dataloader=test_loader, num_bags=3)

        self.save_model(self.model, hash_info, path)

    def audit_model(self, model, dataloader, num_bags=3):
        """
        Audita o modelo treinado, gerando a matriz de confusão e inspecionando 
        os pesos de atenção das instâncias (registos HTTP).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
        import numpy as np
        import torch
        
        logger.info("=== A INICIAR AUDITORIA DO MODELO MIL ===")
        model.eval()
        
        all_preds = []
        all_labels = []
        
        # 1. Avaliação Global (Matriz de Confusão)
        with torch.no_grad():
            for bags, labels in dataloader:
                bags = bags.to(self.device)
                preds, _ = model(bags)
                # Converte probabilidades > 0.5 para a classe 1 (unsafe)
                all_preds.extend((preds > 0.5).cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
        # Gerar Matriz de Confusão
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão MIL - {self.traffic_source}')
        plt.ylabel('Rótulo Real (True Label)')
        plt.xlabel('Previsão do Modelo (Predicted)')
        plt.show() # No Jupyter, isto irá renderizar o gráfico no output da célula
        
        print("\nRelatório de Classificação das Bags:")
        # zero_division=0 evita avisos caso o modelo preveja apenas uma classe
        print(classification_report(all_labels, all_preds, zero_division=0)) 
        
        # 2. Inspeção de Atenção (Auditoria de Instâncias)
        logger.info("=== A INSPECIONAR ATENÇÃO NAS INSTÂNCIAS ===")
        with torch.no_grad():
            # Pega num único lote (batch) para inspecionar
            bags, bag_labels = next(iter(dataloader))
            bags = bags.to(self.device)
            preds, attentions = model(bags)
            
            for b_idx in range(min(num_bags, bags.size(0))):
                bag_pred = preds[b_idx].item()
                bag_true = bag_labels[b_idx].item()
                pesos = attentions[b_idx].squeeze(-1).cpu().numpy()
                
                print(f"\n--- Bag {b_idx} ---")
                print(f"Previsão da Bag: {bag_pred:.4f} (Real: {bag_true})")
                
                # Ordena os pesos para encontrar os maiores e menores
                top_3_indices = pesos.argsort()[::-1][:3]
                bottom_2_indices = pesos.argsort()[:2]
                
                print("🔴 Top 3 Instâncias (Maior suspeita / Foco do modelo):")
                for idx in top_3_indices:
                    print(f"   Registo ID {idx:02d} | Peso de Atenção: {pesos[idx]:.4f}")
                    
                print("🟢 Bottom 2 Instâncias (Considerado ruído / Tráfego normal):")
                for idx in bottom_2_indices:
                    print(f"   Registo ID {idx:02d} | Peso de Atenção: {pesos[idx]:.4f}")

    def save_model(self, model, hash_info, path):
        # Simplificado para salvar apenas o estado do MIL
        save_path = f"{path}/{self.traffic_source}/{self.emb_config}/attention_mil.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        artifact = {
            "model_state_dict": model.state_dict(),
            "config": {"embed_dim": self.emb_dim, "bag_size": self.bag_size},
            "training_info": hash_info
        }
        
        torch.save(artifact, save_path)
        logger.info(f"Modelo salvo em: {save_path}")

    def plot_attention_weights(self, model, dataloader, num_batches_to_plot=1, save_path="attention_plot.png"):
        """
        Gera um Heatmap dos pesos de atenção para inspecionar o que o modelo considerou ruído.
        """
        logger.info("Gerando plot de inspeção visual da atenção...")
        model.eval()
        
        all_attentions = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, (bags, labels) in enumerate(dataloader):
                if i >= num_batches_to_plot:
                    break
                    
                bags = bags.to(self.device)
                preds, A = model(bags)
                
                # A tem o formato (Batch, Bag_Size, 1). Vamos remover a última dimensão.
                # attention_matrix ficará com o formato (Batch, Bag_Size)
                attention_matrix = A.squeeze(-1).cpu().numpy() 
                
                all_attentions.append(attention_matrix)
                all_preds.extend((preds > 0.5).cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        # Concatena todos os batches coletados
        attentions_concat = np.vstack(all_attentions)
        
        # Criação do Plot
        plt.figure(figsize=(12, 8))
        
        # Usamos o Seaborn para criar o heatmap
        ax = sns.heatmap(
            attentions_concat, 
            cmap="viridis", # "viridis" ou "magma" são ótimos para destacar valores altos
            cbar_kws={'label': 'Peso de Atenção (Importância)'},
            xticklabels=True,
            yticklabels=False
        )
        
        # Configurando os eixos
        plt.title(f"Mapa de Atenção MIL - Fonte: {self.traffic_source}", fontsize=14, pad=20)
        plt.xlabel("Instâncias dentro da Bag (Ex: Logs HTTP)", fontsize=12)
        plt.ylabel("Bags (Pacotes de Dados)", fontsize=12)
        
        # Adiciona no eixo Y se a bag era Realmente Positiva e a Previsão do Modelo
        yticks_labels = [f"Real: {int(l)} | Pred: {int(p)}" for l, p in zip(all_labels, all_preds)]
        ax.set_yticks(np.arange(len(yticks_labels)) + 0.5)
        ax.set_yticklabels(yticks_labels, rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Plot salvo com sucesso em: {save_path}")

    async def close_connections(self):
        await self.repo_campaigns.close()
        await self.repo_request.close()