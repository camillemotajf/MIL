from src.training_models.repositories.campaigns_repository import CampaignRepository
from src.training_models.services.campaign_service import CampaignService
from src.training_models.config.database import AsyncSessionLocal

from src.training_models.config.mongo import mongo_collection
from src.training_models.repositories.request_repository import RequestsRepository
from src.training_models.services.request_service import RequestService


campaign_repository = CampaignRepository(
    session_factory=AsyncSessionLocal
)

campaign_service = CampaignService(
    repository=campaign_repository
)

mongo_request_repository = RequestsRepository(
    collection=mongo_collection
)

request_service = RequestService(
    repository=mongo_request_repository
)


