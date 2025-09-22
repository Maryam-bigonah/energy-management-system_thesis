from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data.db")
    if database_url.startswith("sqlite"):
        # Ensure directory exists for file-based SQLite
        path = database_url.replace("sqlite:///", "")
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    return database_url


def create_db_engine() -> Engine:
    return create_engine(get_database_url(), future=True)


engine: Engine = create_db_engine()

app = FastAPI(title="DB Explorer & Model Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
static_dir = os.path.abspath(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


@app.get("/api/tables")
async def list_tables() -> Dict[str, Any]:
    try:
        inspector = inspect(engine)
        tables: List[str] = inspector.get_table_names()
        views: List[str] = inspector.get_view_names()
        return {"tables": tables, "views": views}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/table/{table_name}")
async def preview_table(table_name: str, limit: int = 50) -> Dict[str, Any]:
    try:
        inspector = inspect(engine)
        valid_tables = set(inspector.get_table_names())
        if table_name not in valid_tables:
            raise HTTPException(status_code=404, detail="Table not found")

        with engine.connect() as conn:
            query = text(f"SELECT * FROM \"{table_name}\" LIMIT :limit")
            result = conn.execute(query, {"limit": limit})
            rows = [dict(row._mapping) for row in result]
            columns = list(rows[0].keys()) if rows else []
            return {"columns": columns, "rows": rows}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
