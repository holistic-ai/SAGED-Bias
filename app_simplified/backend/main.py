from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app_simplified.backend.routers import benchmark, database
from app_simplified.backend.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Simplified App API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(benchmark.router)
app.include_router(database.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Simplified App API"} 