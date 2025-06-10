# Source Processing Flow

This document outlines the complete flow of how source files are processed in the SAGED system, from frontend upload to final processing by SAGED.

## 1. Frontend Upload Process

### Source Selection Component
- Located in `app_simplified/frontend/src/components/build_forms/SourceSelection.tsx`
- Handles file uploads and concept assignments
- Key features:
  - File upload interface for .txt files
  - Concept assignment for each uploaded source
  - Toggle between Wikipedia and local file sources
  - Configuration of source finder settings

### Upload Flow
1. User selects files through the frontend interface
2. Files are sent to the backend via POST request to `/files/upload/{domain}`
3. Each file is processed individually and assigned to concepts
4. The configuration is updated with file paths and concept assignments

## 2. Backend Processing

### File Service
- Located in `app_simplified/backend/services/file_service.py`
- Handles file storage and database operations
- Key functions:
  - `save_uploaded_file`: Saves file content to database
  - `get_file_content`: Retrieves file content from database
  - `get_domain_files`: Lists all files for a domain

### Database Service
- Located in `app_simplified/backend/services/database_service.py`
- Manages database operations and table creation
- Key features:
  - Creates domain-specific source text tables
  - Manages database connections and configurations
  - Handles table naming conventions

### Database Structure
- Each domain has its own source text table: `{domain}_source_texts`
- Table schema:
  ```sql
  CREATE TABLE {domain}_source_texts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_path TEXT NOT NULL UNIQUE,
      content TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
  ```

## 3. SAGED Integration

### Configuration Flow
1. Frontend sends configuration to backend
2. Backend's `SagedService` processes the configuration:
   - Updates database configuration
   - Sets source text table name
   - Configures source finder settings
   - Sets up manual sources list

### Source Processing in SAGED
1. **Source Finding Phase**:
   - SAGED reads the configuration
   - For local files:
     - Retrieves file paths from `manual_sources`
     - Uses `SAGEDData.read_text_content` to read content from database
   - For Wikipedia:
     - Uses Wikipedia API to fetch content

2. **Scraping Phase**:
   - Uses `Scraper` class to process sources
   - For local files:
     - Reads content from database using configured table
     - Processes content based on scraping configuration
   - Stores results in specified database tables

## 4. Data Flow Summary

1. **Upload Phase**:
   ```
   Frontend -> Backend API -> File Service -> Database
   ```

2. **Configuration Phase**:
   ```
   Frontend Config -> Backend -> SagedService -> SAGED
   ```

3. **Processing Phase**:
   ```
   SAGED -> Database (read) -> Processing -> Database (write)
   ```

## 5. Key Components

### Frontend
- `SourceSelection.tsx`: Main component for source management
- Handles file uploads and concept assignments
- Manages source finder configuration

### Backend
- `FileService`: Manages file operations
- `DatabaseService`: Handles database operations
- `SagedService`: Integrates with SAGED

### Database
- Domain-specific source text tables
- Stores file content and metadata
- Manages source references

### SAGED
- `SAGEDData`: Handles data operations
- `Scraper`: Processes source content
- Configuration-driven processing pipeline

## 6. Configuration Structure

The source processing configuration includes:

```typescript
{
  shared_config: {
    source_finder: {
      require: true,
      method: 'local_files' | 'wiki',
      manual_sources: string[],
      saving_location: string
    }
  },
  concept_specified_config: {
    [concept: string]: {
      source_finder: {
        manual_sources: string[]
      }
    }
  }
}
```

## 7. Type System Implementation

### Database Configuration Types
```typescript
// Frontend (saged_config.ts)
interface DatabaseConfig {
    use_database: boolean;
    database_type: string;
    database_connection: string;
    table_prefix: string;
    source_text_table: string;
}

# Backend (build_config.py)
class DatabaseConfig(BaseModel):
    use_database: bool = True
    database_type: str = "sql"
    database_connection: str = "sqlite:///./data/db/saged_app.db"
    table_prefix: str = ""
    source_text_table: str = "source_texts"
```

### Source Finder Configuration Types
```typescript
// Frontend (saged_config.ts)
interface SourceFinderConfig {
    require: boolean;
    reading_location: string;
    method: string;
    local_file?: string;
    scrape_number: number;
    saving: boolean;
    saving_location: string;
    scrape_backlinks: number;
    manual_sources?: string[];
}

# Backend (build_config.py)
class SourceFinderConfig(BaseModel):
    require: bool = True
    reading_location: str = "default"
    method: str = "wiki"
    local_file: Optional[str] = None
    scrape_number: int = 5
    saving: bool = True
    saving_location: str = "default"
    scrape_backlinks: int = 0
    manual_sources: Optional[List[str]] = None
```

### Database Models
```python
# Backend (benchmark.py)
class SourceFinder(Base):
    __tablename__ = "source_finder"
    
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    concept = Column(String, index=True)
    concept_shared_source = Column(JSON)  # List of source dictionaries
    keywords = Column(JSON)  # Dictionary of keywords and metadata
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 8. Implementation Notes

### Type Consistency
- Frontend uses TypeScript interfaces for type safety
- Backend uses Pydantic models for validation
- SAGED uses Python type hints and schema validation
- Database models use SQLAlchemy for ORM

### Default Values
- Frontend: `use_database: false`
- Backend: `use_database: true`
- SAGED: `use_database: false`
- Note: These inconsistencies should be aligned in future updates

### Optional Fields
- Frontend: Uses TypeScript's `?` syntax
- Backend: Uses Pydantic's `Optional` type
- SAGED: Uses `None` as default values

### Method Types
- Frontend: Specific string literals for methods
- Backend: Generic string types
- SAGED: Schema-based validation

### Data Flow Type Safety
1. **Frontend to Backend**:
   - TypeScript interfaces ensure correct data structure
   - API endpoints validate incoming data
   - Pydantic models provide runtime validation

2. **Backend to Database**:
   - SQLAlchemy models ensure database schema compliance
   - JSON fields store complex data structures
   - Type conversion handled by ORM

3. **Backend to SAGED**:
   - Configuration validated against schema
   - Data structures mapped to SAGED types
   - Results validated before storage

This configuration structure ensures proper source assignment and processing throughout the pipeline. 