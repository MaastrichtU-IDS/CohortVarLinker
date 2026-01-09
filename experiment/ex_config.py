from dataclasses import dataclass, field
from typing import List, Set, Dict
import os
@dataclass(frozen=True)
class Settings:
    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.8
    
    # Text hints for logic
    DATE_HINTS: List[str] = field(default_factory=lambda: ["visit date", "date of visit", "date of event"])
    
    # Categories suitable for cross-category matching
    CROSS_CATS: Set[str] = field(default_factory=lambda: {
        "measurement", "observation", "condition_occurrence", 
        "condition_era", "observation_period"
    })
    
    # External Resources
    GRAPH_REPO: str = "https://w3id.org/CMEO/graph"

    
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