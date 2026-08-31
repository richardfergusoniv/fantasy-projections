"""Versioned API router."""

from fastapi import APIRouter

from src.app.api.v1 import assistant, auth, dynasty, jobs, leagues, operations, players, projections, trades

api_v1_router = APIRouter()
api_v1_router.include_router(auth.router, tags=["auth"])
api_v1_router.include_router(leagues.router, tags=["leagues"])
api_v1_router.include_router(dynasty.router, tags=["dynasty"])
api_v1_router.include_router(operations.router, tags=["operations"])
api_v1_router.include_router(projections.router, tags=["projections"])
api_v1_router.include_router(trades.router, tags=["trades"])
api_v1_router.include_router(players.router, tags=["players"])
api_v1_router.include_router(assistant.router, tags=["assistant"])
api_v1_router.include_router(jobs.router, tags=["jobs"])
