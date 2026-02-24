from training_models.repositories.request_repository import RequestsRepository
from typing import Tuple, List, Dict

class RequestService:
    def __init__(self, repository: RequestsRepository):
        self.repository = repository
    
    async def fetch_training_sample_by_hashes(
        self,
        hashes: list[str],
        limit_each: int = 10000
    ) -> Tuple[List[Dict], Dict]:
        
        if not hashes:
            raise ValueError("")
        
        results, hashes_info = await self.repository.get_training_sample_by_hashes(
            hashes=hashes, 
            limit_each=limit_each
        )

        return results, hashes_info
    