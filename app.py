import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


_APP_DIR = Path(__file__).resolve().parent
# Optional: create .env with OLLAMA_HOST and OLLAMA_MODEL. If missing, defaults match a local Ollama install.
load_dotenv(_APP_DIR / ".env")

SAMPLE_CSV = """id,customer,amount,region,signup_date
1,Northwind LLC,1200.50,EU,2024-01-05
2,Acme Co,890.00,US,2024-02-11
3,Northwind LLC,1200.50,EU,2024-01-05
4,,450.25,US,2024-03-01
5,Beta Inc,2100.00,APAC,2024-04-18
6,Gamma Ltd,75.99,EU,2024-05-22
"""

app = FastAPI(title="Qualata API", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


class AnalyseTextRequest(BaseModel):
    csv_text: str
    key_columns: list[str] | None = None
    run_ai_summary: bool = True


class OllamaInsightsService:
    """Calls a local Llama (or any) model via Ollama's HTTP API — no OpenAI."""

    def __init__(self) -> None:
        self.host = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")
        self.model = (os.getenv("OLLAMA_MODEL") or "llama3.2").strip()
        try:
            self.timeout_sec = float(os.getenv("OLLAMA_TIMEOUT") or "90")
        except ValueError:
            self.timeout_sec = 90.0

    def summarize(self, payload: dict[str, Any]) -> tuple[str, str]:
        """Returns (summary_text, source) where source is ollama or fallback."""
        if not self.model:
            return self._fallback_summary(payload), "fallback"

        slim = {
            "profile": payload.get("profile"),
            "quality_score": payload.get("quality_score"),
            "issues": payload.get("issues"),
            "duplicate_report": payload.get("duplicate_report"),
            "key_facts": payload.get("key_facts"),
        }
        prompt = (
            "You are a senior data analyst. Given this dataset analysis (CSV or JSON), write a clear, concise summary:\n"
            "STRICT RULES:\n"
            "- Do NOT use markdown\n"
            "- Do NOT use asterisks (*), hashes (#)\n"
            "- Use simple sentences separated by line breaks\n"
            "- formatting characters\n"
            "- Use titles and subheadings to structure the summary\n"
            "1) Brief overview of the dataset\n"
            "2) Main data quality concerns\n"
            "3) What duplicate findings suggest\n"
            "4) 2–3 concrete next steps\n\n"
            f"{json.dumps(slim, ensure_ascii=False, default=str)}"
        )

        
        # prompt = ("""
        #     You are a data quality engine. Be concise and structured.

        #     Return ONLY this format:

        #     OVERVIEW:
        #     <1 sentence with dataset size + quality score>

        #     QUALITY:
        #     - Score: <final_score>/100
        #     - Biggest issue: <single most critical issue>

        #     ISSUES:
        #     - <max 3 bullet points, ranked by severity>

        #     DUPLICATES:
        #     - Exact duplicates: <count> (<rate>%)
        #     - Key duplicates: <count or none>

        #     ACTIONS:
        #     - <max 3 specific, practical actions>

        #     RULES:
        #     - No fluff
        #     - No generic advice
        #     - No repetition
        #     - Max 120 words total

        #     DATA:
        #     {json.dumps(slim, ensure_ascii=False, default=str)}
        #     """
        # )

        

        content = self._ollama_chat(prompt)
        text = (content or "").strip()
        if text:
            return text, "ollama"
        return self._fallback_summary(payload), "fallback"

    def _ollama_chat(self, user_prompt: str) -> str:
        url = f"{self.host}/api/chat"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": user_prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
            return ""
        msg = data.get("message") or {}
        return str(msg.get("content") or "")

    def _fallback_summary(self, payload: dict[str, Any]) -> str:
        profile = payload["profile"]
        quality = payload["quality_score"]
        duplicate = payload["duplicate_report"]
        issues = payload["issues"]
        key_issues = issues[:3] if issues else ["No major issues detected."]
        return (
            f"Dataset has {profile['rows']} rows and {profile['columns']} columns. "
            f"Overall quality score is {quality['final_score']}/100. "
            f"Exact duplicate rows: {duplicate['exact_duplicate_count']} "
            f"({duplicate['exact_duplicate_rate_pct']}%). "
            f"Top concerns: {'; '.join(key_issues)}. "
            "This summary uses built-in analysis only (no AI model). "
            "To enable the Llama/Ollama narrative, start Ollama locally and set OLLAMA_MODEL to a pulled model from `ollama list`."
        )


ai_service = OllamaInsightsService()


def _read_csv_from_upload(file: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV file: {exc}") from exc


def _read_csv_from_text(csv_text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse pasted CSV text: {exc}") from exc


def _dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    null_pct = (df.isnull().mean() * 100).round(2).to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "null_pct_by_column": null_pct,
    }


def _quality_score(df: pd.DataFrame) -> dict[str, float]:
    avg_null = df.isnull().mean().mean()
    completeness = max(0.0, 100 - (avg_null * 100))
    dup_rate = float(df.duplicated().mean()) if len(df) else 0.0
    uniqueness = max(0.0, 100 - (dup_rate * 100))

    numeric_cols = df.select_dtypes(include="number").columns
    outlier_penalty = 0.0
    for col in numeric_cols:
        std = df[col].std()
        if pd.isna(std) or std == 0:
            continue
        z = (df[col] - df[col].mean()) / std
        if float((z.abs() > 3).mean()) > 0.01:
            outlier_penalty += 8

    validity = max(0.0, 100 - outlier_penalty)
    final_score = completeness * 0.4 + uniqueness * 0.35 + validity * 0.25
    return {
        "final_score": round(final_score, 2),
        "completeness": round(completeness, 2),
        "uniqueness": round(uniqueness, 2),
        "validity": round(validity, 2),
    }


def _detect_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for col, pct in (df.isnull().mean() * 100).items():
        if pct >= 20:
            issues.append(f"{col}: {pct:.1f}% missing values")

    dup_rate = float(df.duplicated().mean() * 100) if len(df) else 0.0
    if dup_rate >= 5:
        issues.append(f"{dup_rate:.1f}% exact duplicate rows detected")

    for col in df.select_dtypes(include="number").columns:
        std = df[col].std()
        if pd.isna(std) or std == 0:
            issues.append(f"{col}: constant values")
            continue
        z = (df[col] - df[col].mean()) / std
        outlier_rate = float((z.abs() > 3).mean() * 100)
        if outlier_rate > 1:
            issues.append(f"{col}: {outlier_rate:.1f}% potential outliers")
    return issues


def _duplicate_report(df: pd.DataFrame, key_columns: list[str] | None = None) -> dict[str, Any]:
    exact_count = int(df.duplicated().sum())
    exact_rate = round((exact_count / len(df)) * 100, 2) if len(df) else 0.0

    key_dup_count = None
    key_dup_rate = None
    sample_key_duplicates: list[dict[str, Any]] = []

    if key_columns:
        missing = [col for col in key_columns if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid key_columns; missing columns: {missing}",
            )

        key_dup_mask = df.duplicated(subset=key_columns, keep=False)
        key_dup_count = int(key_dup_mask.sum())
        key_dup_rate = round((key_dup_count / len(df)) * 100, 2) if len(df) else 0.0
        if key_dup_count:
            sample_key_duplicates = (
                df.loc[key_dup_mask, key_columns]
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(5)
                .to_dict(orient="records")
            )

    return {
        "exact_duplicate_count": exact_count,
        "exact_duplicate_rate_pct": exact_rate,
        "key_columns": key_columns or [],
        "key_duplicate_count": key_dup_count,
        "key_duplicate_rate_pct": key_dup_rate,
        "sample_key_duplicate_groups": sample_key_duplicates,
    }


def _top_facts(df: pd.DataFrame) -> list[str]:
    facts: list[str] = []
    for col in df.select_dtypes(include="number").columns[:3]:
        facts.append(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}")
    for col in df.select_dtypes(include="object").columns[:3]:
        if df[col].nunique(dropna=True) > 0:
            top = df[col].value_counts(dropna=True).head(1)
            if not top.empty:
                facts.append(f"{col}: most common '{top.index[0]}' ({int(top.iloc[0])} rows)")
    return facts[:6]


def _analyse_dataframe(
    df: pd.DataFrame, key_columns: list[str] | None = None, run_ai_summary: bool = True
) -> dict[str, Any]:
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    profile = _dataset_profile(df)
    quality_score = _quality_score(df)
    issues = _detect_issues(df)
    duplicate_report = _duplicate_report(df, key_columns)
    key_facts = _top_facts(df)

    result: dict[str, Any] = {
        "profile": profile,
        "quality_score": quality_score,
        "issues": issues,
        "duplicate_report": duplicate_report,
        "key_facts": key_facts,
        "ollama": {
            "host": ai_service.host,
            "model": ai_service.model,
        },
    }

    if run_ai_summary:
        summary, src = ai_service.summarize(result)
        result["ai_summary"] = summary
        result["llm_source"] = src
    else:
        result["ai_summary"] = None
        result["llm_source"] = "disabled"

    return result


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "qualata-api",
        "llm": "ollama",
        "ollama_model": ai_service.model,
        "ui": "/ui",
        "sample_csv": "/sample/csv",
    }


@app.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    path = _APP_DIR / "static" / "index4.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="UI not found (static/index4.html).")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/sample/csv", response_class=PlainTextResponse)
def sample_csv() -> PlainTextResponse:
    return PlainTextResponse(SAMPLE_CSV, media_type="text/csv; charset=utf-8")


@app.post("/analyse/upload")
async def analyse_upload(
    file: UploadFile = File(...),
    key_columns: str | None = None,
    run_ai_summary: bool = True,
) -> dict[str, Any]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    parsed_keys = [x.strip() for x in key_columns.split(",")] if key_columns else None
    df = _read_csv_from_upload(file)
    return _analyse_dataframe(df, parsed_keys, run_ai_summary)


@app.post("/analyse/text")
def analyse_text(request: AnalyseTextRequest) -> dict[str, Any]:
    df = _read_csv_from_text(request.csv_text)
    return _analyse_dataframe(df, request.key_columns, request.run_ai_summary)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
