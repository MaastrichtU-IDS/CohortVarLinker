
# import os
# import logging
# from dataclasses import dataclass, field

# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv(".env")


# # NOTE: using dataclass instead of pydantic due to dependency conflict with decentriq_platform preventing to use pydantic v2
# @dataclass
# class Settings:

#     sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7879"))
#     similarity_threshold  : float =  0.8
#     # logfile_path: str = field(default_factory=lambda: os.getenv("LOGFILE_PATH", "./logs.log"))
#     scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))
#     # jwt_secret: str = field(
#     #     default_factory=lambda: os.getenv("JWT_SECRET", "vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM")
#     # )

  
#     admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

#     data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))
#     # dev_mode: bool = field(default_factory=lambda: os.getenv("DEV_MODE", "false").lower() == "true")

#     @property
#     # def redirect_uri(self) -> str:
#     #     if self.api_host.startswith("localhost"):
#     #         return f"http://{self.api_host}/cb"
#     #     else:
#     #         return f"https://{self.api_host}/cb"

#     @property
#     def auth_audience(self) -> str:
#         # if self.dev_mode:
#         #     return "https://other-ihi-app"
#         # else:
#         return "https://explorer.icare4cvd.eu"

#     @property
#     def query_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/query"

#     @property
#     def update_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/update"

#     @property
#     def authorization_endpoint(self) -> str:
#         return f"{self.auth_endpoint}/authorize"

#     @property
#     def admins_list(self) -> list[str]:
#         return self.admins.split(",")

#     @property
#     def logs_filepath(self) -> str:
#         return os.path.join(settings.data_folder, "logs.log")
    
#     @property
#     def sqlite_db_filepath(self) -> str:
#         return "vocab.db"
    
#     @property
#     def vector_db_path(self) -> str:
#         # return  "komal.qdrant.137.120.31.148.nip.io"
#         return  "localhost"
#     @property
#     def concepts_file_path(self) -> str:
#         # return  "komal.qdrant.137.120.31.148.nip.io"
#         return  "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship.csv"


# settings = Settings()

# # Disable uvicorn logs, does not seems to really do much
# # uvicorn_error = logging.getLogger("uvicorn.error")
# # uvicorn_error.disabled = True
# # uvicorn_access = logging.getLogger("uvicorn.access")
# # uvicorn_access.disabled = True

# # logging.basicConfig(filename=settings.logs_filepath, level=logging.INFO, format="%(asctime)s - %(message)s")

from dataclasses import dataclass, field
from typing import List, Set, Dict
import os
@dataclass(frozen=True)
class Settings:
    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.85
    
    # Text hints for logic
    DATE_HINTS: List[str] = field(default_factory=lambda: ["visit date", "date of visit", "date of event"])
    
    # Categories suitable for cross-category matching
    CROSS_CATS: Set[str] = field(default_factory=lambda: {
        "measurement", "observation", "condition_occurrence", 
        "condition_era", "observation_period"
    })
    
    # External Resources
    GRAPH_REPO: str = "https://w3id.org/CMEO/graph"

    DATA_DOMAINS: List[str] = field(default_factory=lambda: ["drug_exposure","condition_occurrence","condition_era","observation","observation_era","measurement","visit_occurrence","procedure_occurrence","device_exposure","person"])
    # Derived Variable Rules (Example)
    DERIVED_VARIABLES: List[Dict] = field(default_factory=lambda: [
        {
            "name": "BMI-derived",
            "omop_id": 3038553,           
            "code": "loinc:39156-5",
            "label": "Body mass index (BMI) [Ratio]",
            "required_omops": [3016723, 3025315],
            "category": "measurement"
        }
    ])
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

    data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))
    sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7879"))

    EMBEDDING_MODEL_NAME: str = "sapbert"
    @property
    def auth_audience(self) -> str:
        # if self.dev_mode:
        #     return "https://other-ihi-app"
        # else:
        return "https://explorer.icare4cvd.eu"

    @property
    def query_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/query"

    @property
    def update_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/update"

    @property
    def authorization_endpoint(self) -> str:
        return f"{self.auth_endpoint}/authorize"

    @property
    def admins_list(self) -> list[str]:
        return self.admins.split(",")

    @property
    def logs_filepath(self) -> str:
        return os.path.join(settings.data_folder, "logs.log")
    
    @property
    def sqlite_db_filepath(self) -> str:
        return "vocab.db"
    
    @property
    def vector_db_path(self) -> str:
        # return  "komal.qdrant.137.120.31.148.nip.io"
        return  "localhost"
    @property
    def concepts_file_path(self) -> str:
        # return  "komal.qdrant.137.120.31.148.nip.io"
        return  "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship.csv"

settings = Settings()