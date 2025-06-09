from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app_simplified.backend.routers import benchmark, database
from app_simplified.backend.database import engine, Base
from app_simplified.backend.routers.files import router as files_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="SAGED API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(benchmark.router)
app.include_router(database.router)
app.include_router(files_router)

@app.get("/")
async def root():
    return {"message": "Welcome to SAGED API"} 