import json, boto3, traceback
from .logic import run_portfolio_analysis , prepare_dashboard_data, execute_trades
import pandas as pd
import datetime 
import numpy as np
import pandas as pd
import os

def clean_nans(obj):
    """Convierte NaN e infinitos en None dentro de estructuras anidadas."""
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj
def convert_to_serializable(obj):
    """Convierte DataFrames, Series y Numpy arrays a tipos serializables."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

def default_serializer(obj):
    """Convierte objetos no serializables (como date o Timestamp) a string."""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")
    
s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

def handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        job_id = body["job_id"]
        type = body["type"]
        if type == "Analysis":
            try:
                print(f"👉 Procesando job {job_id}")
                # Ejecuta tu análisis
                result = run_portfolio_analysis()
                # Generar datos para dashboard
                portfolio = pd.DataFrame(result["portfolio"])
                df_betas = pd.DataFrame(result["betas"])
                historical_performance = pd.DataFrame(result["historical_performance"])
                returns_by_coin = pd.DataFrame(result["returns_by_coin"])
                print("Return by coins:")
                print(returns_by_coin)
                dashboard_data = prepare_dashboard_data(portfolio, df_betas,historical_performance,returns_by_coin)
                safe_data = clean_nans(dashboard_data)
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"jobs/{job_id}.json",
                    Body=json.dumps({"status": "done", "result": safe_data},default= default_serializer),
                    ContentType="application/json"
                )
                dfs = result.get("dfs", {})
                if isinstance(dfs, dict):
                    for k, v in dfs.items():
                        dfs[k] = pd.DataFrame(v)
                else:
                    raise ValueError("dfs no es un diccionario válido")
                portfolio_table = pd.DataFrame(result["portfolio_table"])
                combined_result = {"portfolio_table": portfolio_table,"dfs": dfs}
                safe_data2 = clean_nans(combined_result)
                serializable_data = convert_to_serializable(safe_data2)
                # Guardar resultado en S3
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"data/{job_id}.json",
                    Body=json.dumps({"status": "done", "result": serializable_data},default= default_serializer),
                    ContentType="application/json"
                )
                print(f"✅ Job {job_id} terminado")
            except Exception as e:
                print("❌ Error procesando job:", e)
                print(traceback.format_exc())
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"jobs/{job_id}.json",
                    Body=json.dumps({"status": "error", "error": str(e)}).encode("utf-8"),
                    ContentType="application/json"
                )
        if type == "Orders":
            try:
                print("Executing trads from job: " + job_id)
                # Ejecutar las ordenes
                obj = s3.get_object(Bucket=BUCKET_NAME,Key=f"data/{job_id}.json")
                body = json.loads(obj["Body"].read().decode("utf-8"))
                print(body)
                result = body.get("result",{})
                portfolio_table = pd.DataFrame(result.get("portfolio_table", []))
                if portfolio_table is None or portfolio_table.empty:
                    raise ValueError("Portfolio/table not loaded")
                dfs = result.get("dfs")
                if "dfs" in result and isinstance(dfs, dict):
                    for k,v in result["dfs"].items():
                        dfs[k] = pd.DataFrame(v)
                else: 
                    raise ValueError("dfs not loaded")
                final_check = execute_trades(dfs,portfolio_table)
                payload = clean_nans({
                    "status":"done",
                    "job_id":job_id,
                    "result":final_check.to_dict(orient="records")})
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key = f"executions/{job_id}.json",
                    Body=json.dumps(payload,default=default_serializer),
                    ContentType="application/json")
                print("Ordenes ejecutadas")
            except Exception as e:
                print("❌ Error procesando job:", e)
                print(traceback.format_exc())
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"jobs/{job_id}.json",
                    Body=json.dumps({"status": "error", "error": str(e)}).encode("utf-8"),
                    ContentType="application/json"
                )
    return {"status": "ok"}
