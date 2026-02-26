import asyncio
from typing import Dict, List, Tuple

class RequestsRepository:
      def __init__(self, collection):
            self.collection = collection

      async def create_indexes(self):
        print("Criando índices do repositório...")
        await self.collection.create_index([
            ("metadata.site", 1),
            ("decision", 1),
        ])      

      async def get_training_sample_by_hashes(self, hashes: list[str], limit_each: int = 10000, rule_id = False) -> Tuple[List[Dict], List[Dict]]:

            projection = {
                  "headers": True,
                  "request": True,
                  "decision": True,
                  "datetime": True,
                  "ip": True,
                  "ip_api_isp": True
            }

            if rule_id:
                  projection["rule_id_list"] = True 

            query_bots = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": ["bots", "bot"]}
            } 

            
            query_unsafe = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": ["unsafe"]}
            }

            pipeline_dates = [
                  {"$match": {"metadata.site": {"$in": hashes}}},
                  {"$group": {
                        "_id": "$metadata.site", 
                        "start_datetime": {"$min": "$datetime"},
                        "end_datetime": {"$max": "$datetime"}
                  }}
            ]

            results = await asyncio.gather(
                  self.collection.find(query_bots, projection).limit(limit_each).sort("datetime", -1).to_list(length=limit_each),
                  self.collection.find(query_unsafe, projection).limit(limit_each).sort("datetime", -1).to_list(length=limit_each),
                  self.collection.aggregate(pipeline_dates).to_list(length=None) # length=None pois o max é o número de hashes
            )

            bots_list = results[0]
            unsafe_list = results[1]
            dates_metadata = results[2]
            print(dates_metadata)

            hash_date_ranges = {
                  item["_id"]: {
                        "start_datetime": item["start_datetime"],
                        "end_datetime": item["end_datetime"]
                  }
                  for item in dates_metadata
            }

            min_count = min(len(bots_list), len(unsafe_list))
            if min_count == 0:
                  return [], hash_date_ranges
            
            final_bots = bots_list[:min_count]
            final_unsafe = unsafe_list[:min_count]

            # Retorna uma Tupla: a lista de amostras e o dicionário com as datas
            return final_bots + final_unsafe, hash_date_ranges



