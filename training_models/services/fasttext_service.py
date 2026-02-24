import os
import logging
import multiprocessing
import pandas as pd
from typing import Tuple, List, Dict
from gensim.models import FastText

from tokenization.http_tokens import create_vocabulary
from training_models.repositories.request_repository import RequestsRepository
from training_models.repositories.campaigns_repository import CampaignRepository

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING) # Mantém o Gensim silencioso

class FastTextTrainingService:
    def __init__(
        self, 
        traffic_source: str, 
        repo_requests: RequestsRepository, 
        repo_campaigns: CampaignRepository
    ):
        self.traffic_source = traffic_source
        self.repo_requests = repo_requests
        self.repo_campaigns = repo_campaigns
        self.limit_cpg: int = 100
        self.limit_each: int = 10000

    async def fetch_training_data(self) -> pd.DataFrame:
        """Busca os dados do MongoDB usando os repositórios existentes."""
        logger.info(f"Buscando campanhas ativas recentes para a fonte: {self.traffic_source}...")
        recent_campaign_hashes = await self.repo_campaigns.get_recent_active_campaign_hashes(
            traffic_source=self.traffic_source,
            limit=self.limit_cpg
        )

        if not recent_campaign_hashes:
            logger.warning("Nenhuma campanha recente encontrada.")
            return pd.DataFrame()
        
        logger.info(f"Buscando amostras de requests para {len(recent_campaign_hashes)} campanhas...")
        results, _ = await self.repo_requests.get_training_sample_by_hashes(
            hashes=recent_campaign_hashes, 
            limit_each=self.limit_each
        )

        if not results:
            logger.warning("Nenhum request encontrado para as campanhas.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        cols_to_keep = [col for col in ["request", "headers", "decision"] if col in df.columns]
        df = df[cols_to_keep]
        
        if "request" in df.columns:
            df = df.rename(columns={"request": "params"})

        return df

    def _ensure_dict_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante que headers e params sejam dicionários."""
        import json
        def safe_loads(s):
            if isinstance(s, dict): return s
            try: return json.loads(s) if pd.notna(s) else {}
            except: return {}

        if 'headers' in df.columns:
            df['headers'] = df['headers'].apply(safe_loads)
        if 'params' in df.columns:
            df['params'] = df['params'].apply(safe_loads)
        return df

    async def _prepare_corpus(self):
        """Método auxiliar para buscar os dados e gerar o corpus."""
        df = await self.fetch_training_data()
        if df.empty:
            return None
        df = self._ensure_dict_format(df)
        logger.info(f"Dados carregados. Shape: {df.shape}")
        
        logger.info("Criando vocabulário (tokenização)...")
        corpus = create_vocabulary(df)
        logger.info(f"Vocabulário criado com {len(corpus)} sentenças.")
        return corpus

    async def train_pipeline(self, vector_size=100, window=10, min_count=1, epochs=10, save_path="models/fasttext_model.model"):
        """Treina um modelo do ZERO e salva."""
        logger.info("=== Iniciando Treinamento FastText do ZERO ===")
        
        corpus = await self._prepare_corpus()
        if not corpus:
            logger.error("Sem dados para treinar. Abortando.")
            return

        cores = multiprocessing.cpu_count()
        workers = max(1, cores - 1)

        model = FastText(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,
            workers=workers
        )

        logger.info("Construindo vocabulário interno inicial...")
        model.build_vocab(corpus_iterable=corpus)

        logger.info(f"Treinando o modelo por {epochs} épocas...")
        model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=epochs)

        # Garante que o diretório existe antes de salvar
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"Modelo criado e salvo em: {save_path}")
        return model

    async def retrain_pipeline(self, save_path="models/fasttext_model.model", epochs=10):
        """Carrega um modelo existente, atualiza o vocabulário e continua o treinamento."""
        logger.info("=== Iniciando RETREINAMENTO Incremental FastText ===")
        
        if not os.path.exists(save_path):
            logger.error(f"Modelo base não encontrado em '{save_path}'. Execute o train_pipeline() primeiro.")
            return None

        corpus = await self._prepare_corpus()
        if not corpus:
            logger.error("Sem novos dados para retreinar. Abortando.")
            return None

        # 1. Carrega o modelo antigo
        logger.info(f"Carregando o modelo salvo em {save_path}...")
        model = FastText.load(save_path)

        # 2. Atualiza o Vocabulário (o update=True é a chave aqui!)
        logger.info("Integrando novas palavras ao vocabulário existente (update=True)...")
        model.build_vocab(corpus_iterable=corpus, update=True)

        # 3. Retreina o modelo
        logger.info(f"Continuando o treinamento por mais {epochs} épocas...")
        model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=epochs)

        # 4. Salva por cima do arquivo antigo
        model.save(save_path)
        logger.info(f"Retreinamento concluído! Modelo atualizado em: {save_path}")
        return model