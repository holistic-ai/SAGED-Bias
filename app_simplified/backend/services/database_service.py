from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, JSON, Float, DateTime, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import pandas as pd
import uuid
from ..schemas.build_config import (
    KeywordsData, SourceFinderData, ScrapedSentencesData,
    ReplacementDescriptionData, BenchmarkData, AllDataTiersResponse
)
from typing import List, Optional
from sqlalchemy.exc import SQLAlchemyError
import logging
import json

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs("data/db", exist_ok=True)
        
        # SQLite database URL with absolute path
        db_path = os.path.abspath("data/db/saged_app.db")
        self.database_url = f"sqlite:///{db_path}"
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False}  # Needed for SQLite
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create metadata
        self.metadata = MetaData()
        
        # Initialize database if not exists
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with required tables if they don't exist"""
        try:
            with self.engine.connect() as conn:
                # Create a test table if it doesn't exist
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS database_status (
                        id INTEGER PRIMARY KEY,
                        status TEXT NOT NULL,
                        last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to initialize database: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if the database is connected"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False
    
    def is_activated(self) -> bool:
        """Check if the database is activated and ready for use"""
        try:
            with self.engine.connect() as conn:
                # Check if the database_status table exists and has a valid status
                result = conn.execute(text("""
                    SELECT status FROM database_status 
                    WHERE id = 1
                """)).first()
                
                if not result:
                    # If no status exists, create one
                    conn.execute(text("""
                        INSERT INTO database_status (id, status) 
                        VALUES (1, 'active')
                    """))
                    conn.commit()
                    return True
                
                return result[0] == 'active'
        except SQLAlchemyError:
            return False
    
    def test_connection(self):
        """Test the database connection with a simple query"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as e:
            raise Exception(f"Database connection test failed: {str(e)}")
    
    def activate_database(self):
        """Activate the database for use"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT OR REPLACE INTO database_status (id, status) 
                    VALUES (1, 'active')
                """))
                conn.commit()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to activate database: {str(e)}")
    
    def deactivate_database(self):
        """Deactivate the database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT OR REPLACE INTO database_status (id, status) 
                    VALUES (1, 'inactive')
                """))
                conn.commit()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to deactivate database: {str(e)}")
    
    def get_database_config(self):
        """Get database configuration for SAGED"""
        return {
            'use_database': True,
            'database_type': 'sql',
            'database_connection': self.database_url
        }
    
    def get_table_name(self, data_tier: str, domain: str) -> str:
        """Get the table name for a specific data tier with domain and unique ID"""
        # Generate a unique ID for this table instance
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        return f"{domain}_{data_tier}_{unique_id}"
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def save_benchmark(self, domain: str, data: dict):
        """Save benchmark data"""
        with self.engine.connect() as conn:
            # Handle both DataFrame and JSON data
            if isinstance(data.get('data'), dict):
                # For JSON data
                conn.execute(
                    f"INSERT INTO {self.get_table_name('benchmark', domain)} (domain, concept, data) VALUES (:domain, :concept, :data)",
                    {
                        'domain': domain,
                        'concept': data.get('concept', 'all'),
                        'data': data
                    }
                )
            else:
                # For DataFrame data
                for _, row in data.get('data', pd.DataFrame()).iterrows():
                    conn.execute(
                        f"INSERT INTO {self.get_table_name('benchmark', domain)} (domain, concept, keyword, prompts, baseline, source_tag) VALUES (:domain, :concept, :keyword, :prompts, :baseline, :source_tag)",
                        {
                            'domain': domain,
                            'concept': row.get('concept', 'all'),
                            'keyword': row.get('keyword'),
                            'prompts': row.get('prompts'),
                            'baseline': row.get('baseline'),
                            'source_tag': row.get('source_tag')
                        }
                    )
            conn.commit()
    
    def get_benchmark(self, domain: str) -> List[BenchmarkData]:
        """Get all benchmark data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('benchmark', domain)} WHERE domain = :domain",
                {'domain': domain}
            ).fetchall()
            return [BenchmarkData(**dict(row)) for row in result]
    
    def get_latest_benchmark(self, domain: str) -> Optional[BenchmarkData]:
        """Get the latest benchmark data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('benchmark', domain)} WHERE domain = :domain ORDER BY created_at DESC LIMIT 1",
                {'domain': domain}
            ).first()
            return BenchmarkData(**dict(result)) if result else None

    def get_keywords(self, domain: str) -> List[KeywordsData]:
        """Get keywords data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('keywords', domain)} WHERE domain = :domain",
                {'domain': domain}
            ).fetchall()
            return [KeywordsData(**dict(row)) for row in result]

    def get_latest_keywords(self, domain: str) -> Optional[KeywordsData]:
        """Get the latest keywords data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('keywords', domain)} WHERE domain = :domain ORDER BY created_at DESC LIMIT 1",
                {'domain': domain}
            ).first()
            return KeywordsData(**dict(result)) if result else None

    def get_source_finder(self, domain: str) -> List[SourceFinderData]:
        """Get source finder data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('source_finder', domain)} WHERE domain = :domain",
                {'domain': domain}
            ).fetchall()
            return [SourceFinderData(**dict(row)) for row in result]

    def get_latest_source_finder(self, domain: str) -> Optional[SourceFinderData]:
        """Get the latest source finder data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('source_finder', domain)} WHERE domain = :domain ORDER BY created_at DESC LIMIT 1",
                {'domain': domain}
            ).first()
            return SourceFinderData(**dict(result)) if result else None

    def get_scraped_sentences(self, domain: str) -> List[ScrapedSentencesData]:
        """Get scraped sentences data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('scraped_sentences', domain)} WHERE domain = :domain",
                {'domain': domain}
            ).fetchall()
            return [ScrapedSentencesData(**dict(row)) for row in result]

    def get_latest_scraped_sentences(self, domain: str) -> Optional[ScrapedSentencesData]:
        """Get the latest scraped sentences data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('scraped_sentences', domain)} WHERE domain = :domain ORDER BY created_at DESC LIMIT 1",
                {'domain': domain}
            ).first()
            return ScrapedSentencesData(**dict(result)) if result else None

    def get_replacement_description(self, domain: str) -> List[ReplacementDescriptionData]:
        """Get replacement description data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('replacement_description', domain)} WHERE domain = :domain",
                {'domain': domain}
            ).fetchall()
            return [ReplacementDescriptionData(**dict(row)) for row in result]

    def get_latest_replacement_description(self, domain: str) -> Optional[ReplacementDescriptionData]:
        """Get the latest replacement description data for a domain"""
        with self.engine.connect() as conn:
            result = conn.execute(
                f"SELECT * FROM {self.get_table_name('replacement_description', domain)} WHERE domain = :domain ORDER BY created_at DESC LIMIT 1",
                {'domain': domain}
            ).first()
            return ReplacementDescriptionData(**dict(result)) if result else None

    def get_all_data_tiers(self, domain: str) -> AllDataTiersResponse:
        """Get all data tiers for a domain"""
        return AllDataTiersResponse(
            keywords=self.get_latest_keywords(domain),
            source_finder=self.get_latest_source_finder(domain),
            scraped_sentences=self.get_latest_scraped_sentences(domain),
            replacement_description=self.get_latest_replacement_description(domain),
            benchmark=self.get_latest_benchmark(domain)
        )

    def save_benchmark_metadata(self, table_name: str, data: dict):
        """Save benchmark metadata to the specified table"""
        try:
            # Create the table if it doesn't exist
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        domain TEXT NOT NULL,
                        data JSON,
                        table_names JSON,
                        configuration JSON,
                        database_config JSON,
                        time_stamp TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Convert dictionary fields to JSON strings
                params = {
                    'domain': data['domain'],
                    'data': json.dumps(data['data']) if data['data'] else None,
                    'table_names': json.dumps(data['table_names']),
                    'configuration': json.dumps(data['configuration']),
                    'database_config': json.dumps(data['database_config']),
                    'time_stamp': data['time_stamp']
                }
                
                # Insert the metadata
                conn.execute(
                    text(f"INSERT INTO {table_name} (domain, data, table_names, configuration, database_config, time_stamp) VALUES (:domain, :data, :table_names, :configuration, :database_config, :time_stamp)"),
                    params
                )
                conn.commit()
                logger.info(f"Successfully saved benchmark metadata to {table_name}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to save benchmark metadata: {str(e)}")
            raise Exception(f"Failed to save benchmark metadata: {str(e)}") 