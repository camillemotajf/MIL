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
from torch.optim import lr_scheduler


# Supondo que essas importações já existam no seu projeto
from data.embedding_service import EmbeddingService
from training_models.repositories.request_repository import RequestsRepository
from training_models.repositories.campaigns_repository import CampaignRepository
import ipaddress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

LABEL_MAP = {"bots": 0, "unsafe": 1}


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

    def __init__(self, df_bags):
        self.bags = []
        self.bag_labels = []
        self.ips = []
        
        for _, row in df_bags.iterrows():
            bag_array = np.array(row["embedding"]) 
            bag_tensor = torch.tensor(bag_array, dtype=torch.float32)

            label_array = np.array(row["bag_label"])
            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            self.bags.append(bag_tensor)
            self.bag_labels.append(label_tensor)
            self.ips.append(row["ip"])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.bag_labels[idx], self.ips[idx]
    


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
        print("Device: ", self.device)
        self.model = None
        self.le = LabelEncoder()
        
        self.get_embedding_instance()
        self.emb_dim = EmbeddingService.get_emb_dim()
        logger.info(f"Dimensao do embedding configurada: {self.emb_dim}")

    async def fetch_training_sample(self) -> Tuple[pd.DataFrame, List[Dict]]:
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
        model_path = f"G:/Meu Drive/TWR/data/{self.traffic_source}/fasttext_{self.traffic_source}.model" if self.emb_config == "fasttext" else "all-MiniLM-L6-v2"
        EmbeddingService.get_instance(self.emb_config, model_path, traffic_source=self.traffic_source)

    def extract_ip_stack(self, ip_string):
        try:
            ip = ipaddress.ip_address(ip_string)
            if ip.version == 4:
                return str(ipaddress.ip_network(f"{ip_string}/24", strict=False))
            elif ip.version == 6:
                return str(ipaddress.ip_network(f"{ip_string}/48", strict=False))
        except ValueError:
            return "ip_invalido"

    async def generate_embeddings(self) -> Tuple[DataLoader, DataLoader, List[Dict]]:
        logger.info("Iniciando geracao de embeddings e Bags MIL por IP...")
        df, hash_info = await self.fetch_training_sample()

        if df.empty:
            raise ValueError(f"Não há amostras suficientes para a fonte '{self.traffic_source}'.")

        df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
        

        mapeamento_mil = {"bots": 1, "unsafe": 0}
        # agrupamento por bloco de ip
        # df["ip_block"] = df["ip"].apply(self.extract_ip_stack)
        # df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
        # #3 bag id -> utilizando pelo MIL Dataset para agrupar
        # df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]

        print(df.head())

        df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
        df["decision_mil"] = df["decision"].map(mapeamento_mil)

        logger.info("Processando e codificando textos do DataFrame...")
        embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
        df["embedding"] = list(embeddings_matrix)

        logger.info("Agrupando requisições em Bags por Endereço IP...")
        bags_df = df.groupby("ip").agg({
            "embedding": list,
            "decision_mil": list,
            # "ip": list
        }).reset_index()

        bags_df["bag_label"] = bags_df["decision_mil"].apply(lambda labels: 1.0 if 1 in labels else 0.0)

        logger.info(f"Criadas {len(bags_df)} Bags lógicas baseadas em IP.")
        print("\nDistribuição Real das Bags (IPs Humanos x IPs de Bots):")
        print(bags_df["bag_label"].value_counts())
        bags_df["bag_size"] = bags_df["embedding"].apply(len)

        # 2. Compara o tamanho médio e a mediana entre Humanos e Bots
        print(bags_df.groupby("bag_label")["bag_size"].describe())
        print("-" * 50)

        df_train, df_test = train_test_split(
            bags_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=bags_df["bag_label"]
        )

        print(df_train.head())

        logger.info("Instanciando os Datasets Lógicos...")
        dataset_train = MILBagDatasetLogical(df_train)
        dataset_test = MILBagDatasetLogical(df_test)

        usar_pin_memory = (self.device == 'cuda')

        train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=usar_pin_memory)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=usar_pin_memory)

        logger.info(f"Gerados {len(dataset_train)} IPs para treino e {len(dataset_test)} IPs para teste.")
        return train_loader, test_loader, hash_info

    async def training_pipeline(self, epochs=10, path="models"):
        train_loader, test_loader, hash_info = await self.generate_embeddings()

        self.model = AttentionMIL(in_features=self.emb_dim, hidden_dim=256).to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss() 

        logger.info("Iniciando treinamento MIL...")

        accumulation_steps = 16
        
        for epoch in range(epochs):
            acumulated_loss = 0.0
            optimizer.zero_grad()

            self.model.train()
            train_loss, train_preds, train_targets = 0.0, [], []
            
            for i, (bags, labels, _) in enumerate(train_loader):
                bags, labels = bags.to(self.device), labels.to(self.device)
                
                preds, _ = self.model(bags)
                loss = criterion(preds, labels)
                loss = loss / accumulation_steps
                
                loss.backward()

                acumulated_loss += loss.item() * accumulation_steps

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()       
                    optimizer.zero_grad()
                
                train_loss += loss.item()
                train_preds.extend((preds > 0.85).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            self.model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []
            
            with torch.no_grad():
                for bags, labels, _ in test_loader:
                    bags, labels = bags.to(self.device), labels.to(self.device)
                    preds, _ = self.model(bags)
                    loss = criterion(preds, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend((preds > 0.85).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            t_acc = accuracy_score(train_targets, train_preds)
            v_acc = accuracy_score(val_targets, val_preds)
            t_loss = train_loss / len(train_loader)
            v_loss = val_loss / len(test_loader)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} - Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} - Acc: {v_acc:.4f}")
        
        self.save_model(self.model, hash_info, path)


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