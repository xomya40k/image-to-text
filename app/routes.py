from fastapi import APIRouter

from .extraction.controllers import extraction_controller


def get_apps_router():
    router = APIRouter()
    router.include_router(extraction_controller.router)
    
    return router