
import sqlite3
import pandas as pd
import datetime
class SQLiteDataManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()
        


    def create_table(self):
        # check if the table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concept_relationship';")
        table_exists = self.cursor.fetchone()
        if table_exists:
            print("Table 'concept_relationship' already exists.")
            # check if the table is empty than add data
            self.cursor.execute("SELECT COUNT(*) FROM concept_relationship;")
            count = self.cursor.fetchone()
            if count[0] == 0:
                self.insert_data_from_csv("/Users/komalgilani/Desktop/chexo_knowledge_graph/concept_relationship.csv")

        else:
            """Create the concept_relationship table if it does not exist."""
            create_table_sql = """
         CREATE TABLE IF NOT EXISTS concept_relationship (
                concept_id_1 INTEGER NOT NULL,
                concept_id_2 INTEGER NOT NULL,
                relationship_id TEXT NOT NULL,
                valid_start_date TEXT NOT NULL,
                valid_end_date TEXT NOT NULL,
                invalid_reason TEXT,
                concept_id_1_name TEXT,
                concept_id_2_name TEXT,
                concept_id_1_vocabulary TEXT NOT NULL,
                concept_id_2_vocabulary TEXT NOT NULL,
                PRIMARY KEY (concept_id_1, concept_id_2, relationship_id)
            );

            """
            self.cursor.execute(create_table_sql)
            self.conn.commit()

            print("Table 'concept_relationship' created successfully.")
            self.insert_data_from_csv("/Users/komalgilani/Desktop/chexo_knowledge_graph/concept_relationship.csv")
    def insert_data_from_csv(self, csv_file):
        """Insert data from CSV into the table, ensuring case normalization."""
        df = pd.read_csv(csv_file, dtype=str, sep='\t')
        print(df.head())

        # Convert column names to lowercase to match table columns
        df.columns = df.columns.str.lower()

        # Convert all string values to lowercase to ensure uniformity
        df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
        df.fillna('"None"', inplace=True)
        # Prepare data for insertion
        to_db = [
            (
                int(row["concept_id_1"]),
                int(row["concept_id_2"]),
                row["relationship_id"],
                row["valid_start_date"],
                row["valid_end_date"],
                row["invalid_reason"] if pd.notna(row["invalid_reason"]) else None,
                row["concept_id_1_name"],
                row["concept_id_2_name"],
                row["concept_id_1_vocabulary"],
                row["concept_id_2_vocabulary"]
            )
            for _, row in df.iterrows()
        ]

        # Insert data into table
        insert_query = """
        INSERT INTO concept_relationship 
        (concept_id_1, concept_id_2, relationship_id, valid_start_date, valid_end_date, invalid_reason, 
        concept_id_1_name, concept_id_2_name, concept_id_1_vocabulary, concept_id_2_vocabulary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.cursor.executemany(insert_query, to_db)
        self.conn.commit()
        print(f"Inserted {len(to_db)} rows into 'concept_relationship'.")

    def check_snomed_relationship(self, concept_id1:int, concept_id2:int, recursion_depth:int=0, max_depth:int=10):
        """
        Check if two concepts are related via SNOMED by following these steps:
        1. If both are SNOMED, check for a direct 'is a' or 'subsumes' relationship.
        2. If one is SNOMED and the other is not, fetch the non-SNOMED concept's SNOMED equivalents 
           (relationship_id includes ' - snomed eq') and recursively check.
        3. If neither is SNOMED, fetch SNOMED equivalents for both and then check if any pair shares a SNOMED relationship.
        """
        # To prevent infinite recursion:

        if recursion_depth > max_depth:
            print("Max recursion depth reached. Stopping further checks.")
            return False, None
        results = self.fetch_relationships(concept_id1, concept_id2)
        if len(results) > 0:
            return True, results
        else:
            # Step 1: Check if both are SNOMED
            if self.is_snomed_concept(concept_id1) and self.is_snomed_concept(concept_id2):
                relationships = self.fetch_relationships(concept_id1, concept_id2)
                # Filter for desired relationships ('is a' or 'subsumes')
                valid_rels = [rel for rel in relationships if rel[2].lower() in ['is a', 'subsumes']]
                if valid_rels:
                    print(f"Found direct relationship: {valid_rels}")
                    return True, valid_rels
                else:
                    # No direct relationship found between the SNOMED concepts.
                    return False, None

            # Step 2: One is SNOMED and the other is not
            
            if self.is_snomed_concept(concept_id1) and not self.is_snomed_concept(concept_id2):
                equivalents = self.fetch_snomed_equivalents(concept_id2)
                for eq in equivalents:
                    found, rels = self.check_snomed_relationship(concept_id1, eq[1], recursion_depth + 1, max_depth)
                    if found:
                        print(f"found relationship between {concept_id1} and {eq[1]}")
                        return True, rels
                return False, None

            if self.is_snomed_concept(concept_id2) and not self.is_snomed_concept(concept_id1):
                equivalents = self.fetch_snomed_equivalents(concept_id1)
                for eq in equivalents:
                    found, rels = self.check_snomed_relationship(eq, concept_id2, recursion_depth + 1, max_depth)
                    if found:
                        print(f"found relationship between {eq[0]} and {concept_id2}")
                        return True, rels
                return False, None

            # Step 3: Neither concept is SNOMED
            if not self.is_snomed_concept(concept_id1) and not self.is_snomed_concept(concept_id2):
                equivalents1 = self.fetch_snomed_equivalents(concept_id1)
                equivalents2 = self.fetch_snomed_equivalents(concept_id2)
                for eq1 in equivalents1:
                    for eq2 in equivalents2:
                        found, rels = self.check_snomed_relationship(eq1, eq2, recursion_depth + 1, max_depth)
                        if found:
                            return True, rels
                return False, None
    def delete_unwanted_relationships(self, retain_relationships=['maps to', 'mapped from', 'is a', 'subsumes']):
        """Delete all relationships except the specified ones."""
        retain_tuple = tuple(rel.lower() for rel in retain_relationships)

        delete_query = f"""
        DELETE FROM concept_relationship
        WHERE LOWER(relationship_id) NOT IN ({','.join(['?'] * len(retain_tuple))});
        """
        self.cursor.execute(delete_query, retain_tuple)
        self.conn.commit()
        print("Deleted unwanted relationships, retaining only:", retain_relationships)



    def bulk_search_relationships(self, concept_id, target_ids):
        """Finds all relationships of a given concept_id with a given set of target IDs."""
        if not target_ids:
            print("No target IDs provided.")
            return []

        # Generate dynamic placeholders for target IDs
        placeholders = ','.join(['?'] * len(target_ids))
        
        # Query to find relationships where concept_id_1 is the given ID, and concept_id_2 is in the target set
        query = f"""
        SELECT concept_id_1, concept_id_2, relationship_id
        FROM concept_relationship
        WHERE (concept_id_1 = ? AND concept_id_2 IN ({placeholders}))
           OR (concept_id_2 = ? AND concept_id_1 IN ({placeholders}));
    """

        params = (concept_id, *target_ids, concept_id, *target_ids)

        # Execute the query with parameters
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        print(results)
        return results
    def fetch_relationships(self, concept_id1, concept_id2):
        """Fetch relationships between two concepts."""
        query = """
                SELECT concept_id_1, concept_id_2, relationship_id
                FROM concept_relationship
                WHERE (concept_id_1 = ? AND concept_id_2 = ?)
                OR (concept_id_1 = ? AND concept_id_2 = ?);
                """
        self.cursor.execute(query, (concept_id1, concept_id2, concept_id2, concept_id1))
        results = self.cursor.fetchall()
        return results
    
    def fetch_concept_relationships(self, concept_id1, relationship_substr='snomed eq'):
        """Fetch relationships between two concepts, matching substring on relationship_id."""
        query = """
                SELECT concept_id_1, concept_id_2, relationship_id
                FROM concept_relationship
                WHERE (concept_id_1 = ? AND relationship_id LIKE ?)
                    OR (concept_id_2 = ? AND relationship_id LIKE ?);
                """
        
        # Construct the substring pattern: e.g. '%snomed eq%'
        substring_pattern = f"%{relationship_substr}%"

        self.cursor.execute(query, (concept_id1, substring_pattern, concept_id1, substring_pattern))
        results = self.cursor.fetchall()

        # remove duplicates
        results = list(set(results))
        return results

    def fetch_snomed_equivalents(self, concept_id):
        """Fetch all SNOMED equivalents of a given concept."""
        query = """
        SELECT concept_id_1, concept_id_2, relationship_id
        FROM concept_relationship
        WHERE (concept_id_1 = ? AND (relationship_id LIKE '% - snomed eq%' or relationship_id='snomed infer'))
        OR (concept_id_2 = ? AND (relationship_id LIKE '% - snomed eq%' or relationship_id='snomed infer'));
        """
        self.cursor.execute(query, (concept_id, concept_id))
        results = self.cursor.fetchall()
        
        return results
    def is_snomed_concept(self, concept_id:int):
        """Check if a concept is a SNOMED concept."""
        query = """
        SELECT concept_id_1, concept_id_2, relationship_id
        FROM concept_relationship
        WHERE (concept_id_1 = ? AND concept_id_1_vocabulary = 'snomed')
        OR (concept_id_2 = ? AND concept_id_2_vocabulary = 'snomed');
        """
        print(type(concept_id))
        self.cursor.execute(query, (concept_id, concept_id))
        results = self.cursor.fetchall()
        return bool(results)

    def fetch_concept_details(self, concept_id):
        # fetch name and vocabulary of the concept
        query = """
        SELECT concept_id_1_name, concept_id_1_vocabulary
        FROM concept_relationship
        WHERE concept_id_1 = ?
        """
        self.cursor.execute(query, (concept_id,))
        results = self.cursor.fetchall()
        return results

        
    def store_inferred_relationship(self, concept_id1, concept_id2, relationship_id="snomed infer"):
        """
        Store an inferred relationship so future lookups can be done quickly.
        This uses INSERT OR IGNORE so that we don't duplicate a row if it already exists.
        """
        # Use some default valid date range, or customize as needed
        valid_start_date = datetime.datetime.now().strftime('%Y-%m-%d')  # "YYYY-MM-DD"

        valid_end_date = "2099-12-31"
        invalid_reason = None

        # In some workflows, you might look up the concept names from a 'concept' table
        # or any other metadata. If you don’t have them, you can store placeholders or omit them.
        concept_d =self.fetch_concept_details(concept_id1)
        concept_id_1_name = concept_d[0][0]
        vocab1 = concept_d[0][1]
        concept_d = self.fetch_concept_details(concept_id2)
        concept_id_2_name = concept_d[0][0]
        vocab2 = concept_d[0][1]


        insert_sql = """
            INSERT OR IGNORE INTO concept_relationship 
            (concept_id_1, concept_id_2, relationship_id,
            valid_start_date, valid_end_date, invalid_reason,
            concept_id_1_name, concept_id_2_name,
            concept_id_1_vocabulary, concept_id_2_vocabulary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            self.cursor.execute(insert_sql, (
                concept_id1,
                concept_id2,
                relationship_id,
                valid_start_date,
                valid_end_date,
                invalid_reason,
                concept_id_1_name,
                concept_id_2_name,
                vocab1,
                vocab2
            ))
            self.conn.commit()
            print(f"[INFO] Inserted or ignored new inferred relationship: {concept_id1} -> {concept_id2} ({relationship_id})")
        except sqlite3.IntegrityError as e:
            print(f"[WARNING] Error inserting inferred relationship: {e}")
    def close_connection(self):
        """Close the database connection."""
        self.cursor.close()
        self.conn.close()
        print("Database connection closed.")

# Example usage
if __name__ == "__main__":
    db_path = "vocab.db"  # SQLite database file
    csv_file = "/Users/komalgilani/Desktop/chexo_knowledge_graph/concept_relationship.csv"  # Path to your CSV file

    # Initialize SQLite manager
    db = SQLiteDataManager(db_path)
    print(db.fetch_concept_relationships(79908))

#     # # Insert data from CSV (Only if it's not already populated)
#     # 

#     # Delete relationships that are not in the desired list
#     # db.delete_unwanted_relationships()

#     # Fetch specific relationships
#     relationships = db.fetch_relationships(4186998, 956874)  # Example concept IDs
#     print("Relationships:", relationships)

#     # Close connection
#     db.close_connection()
