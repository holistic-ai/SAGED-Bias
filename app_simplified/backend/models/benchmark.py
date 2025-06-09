from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Keywords(Base):
    __tablename__ = "keywords"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    keywords = Column(JSON)  # Dictionary of keywords and their metadata
    created_at = Column(DateTime, default=datetime.utcnow)

class SourceFinder(Base):
    __tablename__ = "source_finder"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    concept_shared_source = Column(JSON)  # List of source dictionaries with source_tag, source_type, source_specification
    keywords = Column(JSON)  # Dictionary of keywords and their metadata
    created_at = Column(DateTime, default=datetime.utcnow)

class ScrapedSentences(Base):
    __tablename__ = "scraped_sentences"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    concept_shared_source = Column(JSON)  # List of source dictionaries with source_tag, source_type, source_specification
    keywords = Column(JSON)  # Dictionary of keywords and their metadata with scraped_sentences
    created_at = Column(DateTime, default=datetime.utcnow)

class SplitSentences(Base):
    __tablename__ = "split_sentences"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    keyword = Column(String)
    prompts = Column(String)
    baseline = Column(String)
    source_tag = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Questions(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    keyword = Column(String)
    prompts = Column(String)
    baseline = Column(String)
    source_tag = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReplacementDescription(Base):
    __tablename__ = "replacement_description"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    replacement_description = Column(JSON)  # Dictionary of replacement descriptions
    created_at = Column(DateTime, default=datetime.utcnow)

class Benchmark(Base):
    __tablename__ = "benchmark"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    keyword = Column(String, nullable=True)
    prompts = Column(String, nullable=True)
    baseline = Column(String, nullable=True)
    source_tag = Column(String, nullable=True)
    data = Column(JSON, nullable=True)  # For JSON data
    created_at = Column(DateTime, default=datetime.utcnow) 