from fastapi.staticfiles import StaticFiles
from .logic import run_portfolio_analysis,download_data
import os,threading, time, boto3,json
from datetime import date, datetime
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from uuid import uuid4

s3 = boto3.client("s3")
BUCKET_NAME = "fastapi-bucket-project"

app = FastAPI(title="Portfolio Risk API")

# CORS abierto para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Diccionario en memoria para simular resultados
jobs = {}

@app.get("/start-analysis")
def start_job():
    """Crea un job y lo marca como running en S3"""
    job_id = str(uuid4())

    # Estado inicial en S3
    try:
        s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"jobs/{job_id}.json",
        Body=json.dumps({"status": "running"}).encode("utf-8"),
        ContentType="application/json"
        )
    except Exception as e:
        import traceback
        print("❌ Error en put_object:", e)
        print(traceback.format_exc())
        raise

    # Lanzar análisis en un thread
    threading.Thread(target=run_portfolio, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "status": "running"}

@app.get("/get-result/{job_id}")
def get_result(job_id: str):
    """Consulta el estado de un job en S3"""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"jobs/{job_id}.json")
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except s3.exceptions.NoSuchKey:
        return {"error": "Job not found"}


@app.get("/run_portfolio_analysis")
def run_portfolio(job_id):
    try:
        print(f"Job {job_id} lanzado")
        print("Analyzing..... /")
        result = run_portfolio_analysis()
        print("Analysis ended /")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"jobs/{job_id}.json",
            Body=json.dumps({"status": "done", "result": result}),
            ContentType="application/json"
        )
        print(f"Job {job_id} terminado")
    except Exception as e:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"jobs/{job_id}.json",
            Body=json.dumps({"status": "error", "error": str(e)}),
            ContentType="application/json"
    )
@app.get("/")
def root():
    print("Exexuting...")
    return {"message": "API is running"}
handler = Mangum(app)

# ---------- Helpers de serialización ----------
def _iso(x):
    return x.isoformat() if hasattr(x, "isoformat") else x

def df_to_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    # Limpia NaN -> None y fechas -> str
    out = (
        df.copy()
        .astype(object)
        .where(pd.notnull(df), None)
        .applymap(_iso)
        .to_dict(orient="records")
    )
    return out

@app.get("/ohlc/{coin}")
def ohlc(
    coin: str,
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(3000, ge=1, le=10000),
):
    # Carga datos (puedes cachear en /tmp si quieres acelerar)
    portfolio, dfs = download_data()
    if coin not in dfs:
        raise HTTPException(status_code=404, detail="Coin not found")

    df = dfs[coin].copy()

    # Normaliza fechas si existen
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Filtros
    if start:
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        df = df[df["date"] >= start_d] if "date" in df.columns else df
    if end:
        end_d = datetime.strptime(end, "%Y-%m-%d").date()
        df = df[df["date"] <= end_d] if "date" in df.columns else df

    # Limita filas
    if len(df) > limit:
        df = df.tail(limit)

    return df_to_records(df)

@app.get("/dashboard-data")
def dashboard_data():
    return run_dashboard_data()

@app.get("/coins")
def coins():
    portfolio, dfs = download_data()
    # Preferimos del portfolio para “oficiales”:
    coin_list = sorted(pd.Series(portfolio["Coin"]).dropna().unique().tolist())
    return {"coins": coin_list}




