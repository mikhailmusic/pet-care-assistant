from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger

from app.config import settings
from app.integrations import init_db, close_db
from app.integrations import minio_service
from app.utils.exceptions import PetCareException
from app.api import auth_api, chats_api, messages
from app.agents.calendar_agent import CalendarAgent
from app.dependencies.services import get_user_service
from app.services.user_service import UserService
from fastapi import Depends


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events для FastAPI приложения"""
    # Startup
    logger.info("🚀 Starting PetCare AI Assistant...")
    
    logger.info("Initializing database...")
    await init_db()
    logger.info("✅ Database initialized successfully")
    
    # Проверка MinIO
    logger.info("Checking MinIO connection...")
    try:
        bucket_created = await minio_service.ensure_bucket_exists()
        if bucket_created:
            logger.info(f"✅ MinIO bucket '{minio_service.bucket_name}' created")
        else:
            logger.info(f"✅ MinIO bucket '{minio_service.bucket_name}' already exists")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MinIO: {e}")
        raise
    
    logger.info("✅ Application started successfully")
    
    yield  # Приложение работает
    
    # Shutdown
    logger.info("🛑 Shutting down PetCare AI Assistant...")
    await close_db()
    logger.info("✅ Application stopped")


# Создание FastAPI приложения
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Интеллектуальный ассистент по уходу за домашними животными",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_api.router)
app.include_router(chats_api.router) 
app.include_router(messages.router) 


@app.exception_handler(PetCareException)
async def petcare_exception_handler(request: Request, exc: PetCareException):
    """Обработчик кастомных исключений приложения"""
    logger.error(f"PetCareException: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик необработанных исключений"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "message": "PetCare AI Assistant API",
        "status": "healthy",
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "docs": "/docs" if settings.DEBUG else "Disabled in production",
    }

@app.post("/test/calendar", tags=["Testing"])
async def test_calendar_agent(
    request: dict,  # {"user_id": 123, "message": "...", "context": {...}}
    user_service: UserService = Depends(get_user_service)
):
    """
    Тестовый endpoint для проверки CalendarAgent
    
    Пример запроса:
    {
        "user_id": 1,
        "message": "Напомни о прививке завтра в 14:00",
        "context": {
            "user_timezone": "Europe/Moscow",
            "current_pet_name": "Барсик"
        }
    }
    """
    try:
        # Создаём UserService
        
        
        # Создаём CalendarAgent
        agent = CalendarAgent(user_service=user_service)
        
        # Обрабатываем запрос
        result = await agent.process(
            user_id=request.get("user_id"),
            user_message=request.get("message"),
            context=request.get("context", {})
        )
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.exception("Test calendar agent failed")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    
    if settings.DEBUG:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug",
        )
    else:
        logger.warning("Use 'uvicorn app.main:app' to run in production")