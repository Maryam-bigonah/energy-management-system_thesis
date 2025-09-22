# Thesis DB Explorer & Model Viewer

A minimal full-stack app that:
- Lists database tables/views
- Previews sample rows
- Renders a simple text-based model diagram from uploaded JSON

## Stack
- Backend: FastAPI + SQLAlchemy
- Frontend: Static HTML/CSS/JS served by FastAPI

## Setup

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure database connection (optional):

Set `DATABASE_URL` env var. Defaults to a local SQLite file `./data.db`.

Examples:
- SQLite (default): `sqlite:///./data.db`
- Postgres: `postgresql+psycopg://user:pass@localhost:5432/dbname`
- MySQL: `mysql+pymysql://user:pass@localhost:3306/dbname`

You may need to install the corresponding DB driver, e.g. for Postgres:

```bash
pip install psycopg[binary]
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Model JSON format
A minimal example you can upload in the UI:

```json
{
  "entities": [
    { "name": "users", "fields": ["id", "name", "email"] },
    { "name": "posts", "fields": ["id", "user_id", "title"] }
  ],
  "relations": [
    { "from": "posts", "to": "users", "on": "posts.user_id = users.id" }
  ]
}
```

## Notes
- Table preview escapes the table name and limits to 50 rows.
- For non-SQLite databases, install the appropriate driver.
