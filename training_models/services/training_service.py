import ipaddress
import sys
import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

# Supondo que suas classes MIL estejam nestes caminhos:
from attention_based import AttentionMIL, MILBagDatasetLogical
from data.embedding_service import EmbeddingService
from training_models.repositories.request_repository import RequestsRepository
from training_models.repositories.campaigns_repository import CampaignRepository
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 1. LOGS SINCRONIZADOS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)

# Mapeamento estrito do MIL (Bot = 1, Unsafe/Humano = 0 para BCELoss)
MIL_LABEL_MAP = {"bots": 1, "unsafe": 0}
FASTTEXT_PATH = "G:/Meu Drive/TWR/data"
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

class MILTrainingService:
    def __init__(
        self, 
        traffic_source: str, 
        repo_requests: RequestsRepository, 
        repo_campaigns: CampaignRepository,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        emb_config: str = "fasttext"
    ):
        logger.info("Inicializando MILTrainingService...")
        self.repo_request = repo_requests
        self.repo_campaigns = repo_campaigns
        self.traffic_source = traffic_source
        self.emb_config = emb_config
        self.limit_cpg: int = 100
        self.limit_each: int = 10000
        self.device = device
        
        # O MIL não usa LabelEncoder do sklearn, usamos o mapeamento direto para Float
        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        
        logger.info(f"Dispositivo configurado: {self.device}")
        logger.info("Definindo instância do embedding...")
        self.get_embedding_instance()
        self.emb_dim = self.get_emb_dim()
        logger.info(f"Dimensão do embedding configurada: {self.emb_dim}")

    def get_emb_dim(self):
        return EmbeddingService.get_emb_dim()

    def get_embedding_instance(self):
        if self.emb_config == "fasttext":
            model_path = f"{FASTTEXT_PATH}/{self.traffic_source}/fasttext_{self.traffic_source}.model"
        else:
            model_path = TRANSFORMER_MODEL
        
        logger.info(f"Carregando instância de embedding ({self.emb_config}) do caminho: {model_path}...")
        EmbeddingService.get_instance(self.emb_config, model_path, traffic_source=self.traffic_source)

    async def fetch_training_sample(self) -> Tuple[pd.DataFrame, List[Dict]]:
        logger.info(f"Buscando campanhas ativas recentes para a fonte: {self.traffic_source}...")
        recent_campaign_hashes = await self.repo_campaigns.get_recent_active_campaign_hashes(
            traffic_source=self.traffic_source,
            limit=self.limit_cpg
        )

        if not recent_campaign_hashes:
            logger.warning("Nenhuma campanha recente encontrada.")
            return pd.DataFrame(), {}
        
        logger.info(f"Encontradas {len(recent_campaign_hashes)} campanhas. Buscando amostras de requests...")
        results, hashes_info = await self.repo_request.get_training_sample_by_hashes(
            hashes=recent_campaign_hashes, 
            limit_each=self.limit_each
        )

        logger.info(f"Amostras encontradas: {len(results)}. Convertendo para DataFrame...")
        df = pd.DataFrame(results)

        return df, hashes_info
    
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
        logger.info("Iniciando geração de embeddings e criação de Bags (MIL)...")
        df, hash_info = await self.fetch_training_sample()

        if df.empty:
            raise ValueError(f"Não há amostras suficientes no banco para treinar a fonte '{self.traffic_source}'.")
        
        # agrupamento por bloco de ip
        df["ip_block"] = df["ip"].apply(self.extract_ip_stack)
        df["ip_api_isp"] = df["ip_api_isp"].fillna("ip_unknow")
        #3 bag id -> utilizando pelo MIL Dataset para agrupar
        df["bag_id"] = df["ip_block"] + " | " + df["ip_api_isp"]
        print(df.head())

        df["decision"] = df["decision"].str.lower().replace({"bot": "bots"})
        df["decision_mil"] = df["decision"].map(MIL_LABEL_MAP)

        logger.info("Processando e codificando textos do DataFrame...")
        embeddings_matrix, _ = EmbeddingService.process_and_encode(df)
        df["embedding"] = list(embeddings_matrix)

        logger.info("Agrupando requisições em Bags por Endereço IP...")
        bags_df = df.groupby("bag_id").agg({
            "embedding": list,
            "decision_mil": list,
            "ip": list
        }).reset_index()

        bags_df["bag_label"] = bags_df["decision_mil"].apply(lambda labels: 1.0 if 1 in labels else 0.0)

        logger.info(f"Total de IPs (Bags) formados: {len(bags_df)}")

        logger.info("Dividindo Bags em treino (80%) e teste (20%)...")
        df_train, df_test = train_test_split(
            bags_df, 
            test_size=0.2,     
            random_state=42,    
            stratify=bags_df['bag_label'] 
        )

        self.train_data = df_train
        self.test_data = df_test
        
        logger.info("Criando MILBagDatasetLogical...")
        dataset_train = MILBagDatasetLogical(df_train)
        dataset_test = MILBagDatasetLogical(df_test)

        logger.info("Instanciando DataLoaders (batch_size=1 para MIL)...")
        train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        logger.info("Geração de embeddings e DataLoaders concluída.")
        return train_loader, test_loader, hash_info

    async def training_pipeline(self, epochs=10, hidden_dim=256, lr=1e-3, path=FASTTEXT_PATH):
        logger.info("Iniciando pipeline de treinamento MIL com Atenção...")
        train_loader, test_loader, hash_info = await self.generate_embeddings()

        logger.info("Instanciando a rede neural AttentionMIL...")
        modelo = AttentionMIL(in_features=self.emb_dim, hidden_dim=hidden_dim).to(self.device)
        
        optimizer = AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss() # Binary Cross Entropy (Exige saídas entre 0 e 1)

        logger.info(f"Treinando por {epochs} épocas...")
        
        for epoch in range(epochs):
            modelo.train()
            loss_acumulada = 0.0
            
            for bag, label, _ in train_loader:
                bag, label = bag.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                pred, _ = modelo(bag)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                
                loss_acumulada += loss.item()
                
            loss_media = loss_acumulada / len(train_loader)
            logger.info(f"Época {epoch+1}/{epochs} - Train Loss: {loss_media:.4f}")

        logger.info("Treino concluído. Salvando modelo...")
        self.save_model(modelo, hidden_dim, hash_info, path=path)
        logger.info("Pipeline de treinamento finalizado com sucesso!")

    def save_model(self, modelo, hidden_dim, hash_info, path):
        logger.info("Montando pacote de inferência (AttentionMIL_bundle)...")
        
        inference_artifact = {
            "model_state_dict": modelo.state_dict(),
            "config": {
                "in_features": self.emb_dim,
                "hidden_dim": hidden_dim
            },
            "training_info": hash_info
        }

        save_path = f"{path}/{self.traffic_source}/{self.emb_config}/attention_mil_bundle.pth"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"Salvando o modelo no caminho: {save_path}")
        torch.save(inference_artifact, save_path)
        logger.info("Modelo salvo com sucesso.")
      
    async def close_connections(self):
        logger.info("Fechando conexões com repositórios...")
        await self.repo_campaigns.close()
        await self.repo_request.close()
        logger.info("Conexões fechadas.")