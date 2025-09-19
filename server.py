import json
import os
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from db import Base, engine, get_db
from models import TestCase as TestCaseORM
from utils import detect_domain, generate_ai_test_cases, generate_mock_test_cases

import io
import csv
import pandas as pd

app = FastAPI(title="AI Test Case Generator API", version="1.0.0")

# CORS for local development and static file usage
app.add_middleware(
    CORSMiddleware,
    # Support any origin, including file:// which appears as 'null'
    allow_origins=["*", "null"],
    allow_origin_regex=".*",
    allow_credentials=False,  # must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    user_story: str = Field(..., description="Requirement or user story text")
    domain: Optional[str] = Field(None, description="Domain override: general, embedded, thermal, safety")
    count: int = Field(3, ge=1, le=20)
    use_ai: bool = Field(True)
    model: Optional[str] = Field("llama-3.1-8b-instant")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(2000, ge=256, le=8192)
    extra_instructions: Optional[str] = Field("")
    id_prefix: Optional[str] = Field("TC")


class TestCaseOut(BaseModel):
    id: int
    external_id: Optional[str]
    test_title: str
    description: str
    preconditions: Optional[str]
    test_steps: List[str]
    test_data: Optional[str]
    expected_result: Optional[str]
    priority: Optional[str]
    test_type: Optional[str]
    domain: Optional[str]
    comments: Optional[str]
    source_story: Optional[str]

    # pydantic v2 style (replaces orm_mode)
    model_config = ConfigDict(from_attributes=True)


class TestCaseUpdate(BaseModel):
    external_id: Optional[str] = None
    test_title: Optional[str] = None
    description: Optional[str] = None
    preconditions: Optional[str] = None
    test_steps: Optional[List[str]] = None
    test_data: Optional[str] = None
    expected_result: Optional[str] = None
    priority: Optional[str] = None
    test_type: Optional[str] = None
    domain: Optional[str] = None
    comments: Optional[str] = None


class Paginated(BaseModel):
    items: List[TestCaseOut]
    total: int
    page: int
    page_size: int


class BatchGenerateRequest(BaseModel):
    stories: List[str]
    domain: Optional[str] = None
    count_per_story: int = Field(3, ge=1, le=20)
    use_ai: bool = True
    model: Optional[str] = "llama-3.1-8b-instant"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2000
    extra_instructions: Optional[str] = ""
    id_prefix: Optional[str] = "TC"


class Settings(BaseSettings):
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    # Ignore extra env vars (like GROQ_API_KEY), and load .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # load from env


def check_api_key(x_api_key: Optional[str]):
    if settings.api_key:
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.post("/generate", response_model=List[TestCaseOut])
def generate(req: GenerateRequest, db: Session = Depends(get_db), x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    domain = req.domain or detect_domain(req.user_story)
    if req.use_ai:
        data = generate_ai_test_cases(
            req.user_story,
            domain,
            count=req.count,
            model=req.model or "llama-3.1-8b-instant",
            temperature=req.temperature or 0.2,
            max_tokens=req.max_tokens or 2000,
            extra_instructions=req.extra_instructions or "",
        )
    else:
        data = generate_mock_test_cases(req.user_story, domain, req.count)

    results: List[TestCaseOut] = []
    # persist
    for idx, tc in enumerate(data, start=1):
        external_id = f"{req.id_prefix}-{idx:03d}" if req.id_prefix else None
        steps = tc.get("test_steps") or []
        if not isinstance(steps, list):
            steps = [str(steps)]
        orm = TestCaseORM(
            external_id=external_id,
            test_title=tc.get("test_title") or tc.get("title") or "",
            description=tc.get("description") or "",
            preconditions=tc.get("preconditions"),
            test_steps=json.dumps(steps, ensure_ascii=False),
            test_data=tc.get("test_data"),
            expected_result=tc.get("expected_result"),
            priority=tc.get("priority"),
            test_type=tc.get("test_type"),
            domain=domain,
            comments=tc.get("comments"),
            source_story=req.user_story,
        )
        db.add(orm)
        db.flush()
        results.append(
            TestCaseOut(
                id=orm.id,
                external_id=orm.external_id,
                test_title=orm.test_title,
                description=orm.description,
                preconditions=orm.preconditions,
                test_steps=steps,
                test_data=orm.test_data,
                expected_result=orm.expected_result,
                priority=orm.priority,
                test_type=orm.test_type,
                domain=orm.domain,
                comments=orm.comments,
                source_story=orm.source_story,
            )
        )
    db.commit()
    return results


@app.put("/test-cases/{item_id}", response_model=TestCaseOut)
def update_test_case(item_id: int, payload: TestCaseUpdate, db: Session = Depends(get_db), x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    it = db.get(TestCaseORM, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Not found")
    data = payload.dict(exclude_unset=True)
    if "test_steps" in data and data["test_steps"] is not None:
        data["test_steps"] = json.dumps(data["test_steps"], ensure_ascii=False)
    for k, v in data.items():
        setattr(it, k, v)
    db.add(it)
    db.commit()
    db.refresh(it)
    try:
        steps = json.loads(it.test_steps) if it.test_steps else []
    except Exception:
        steps = []
    return TestCaseOut(
        id=it.id,
        external_id=it.external_id,
        test_title=it.test_title,
        description=it.description,
        preconditions=it.preconditions,
        test_steps=steps,
        test_data=it.test_data,
        expected_result=it.expected_result,
        priority=it.priority,
        test_type=it.test_type,
        domain=it.domain,
        comments=it.comments,
        source_story=it.source_story,
    )


@app.delete("/test-cases/{item_id}")
def delete_test_case(item_id: int, db: Session = Depends(get_db), x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    it = db.get(TestCaseORM, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(it)
    db.commit()
    return {"status": "deleted", "id": item_id}


@app.post("/generate/batch", response_model=List[TestCaseOut])
def generate_batch(req: BatchGenerateRequest, db: Session = Depends(get_db), x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    out: List[TestCaseOut] = []
    domain = req.domain
    for s_idx, story in enumerate(req.stories, start=1):
        d = domain or detect_domain(story)
        data = generate_ai_test_cases(
            story,
            d,
            count=req.count_per_story,
            model=req.model or "llama-3.1-8b-instant",
            temperature=req.temperature or 0.2,
            max_tokens=req.max_tokens or 2000,
            extra_instructions=req.extra_instructions or "",
        ) if req.use_ai else generate_mock_test_cases(story, d, req.count_per_story)

        for idx, tc in enumerate(data, start=1):
            external_id = f"{req.id_prefix}-{s_idx:02d}-{idx:03d}" if req.id_prefix else None
            steps = tc.get("test_steps") or []
            if not isinstance(steps, list):
                steps = [str(steps)]
            orm = TestCaseORM(
                external_id=external_id,
                test_title=tc.get("test_title") or tc.get("title") or "",
                description=tc.get("description") or "",
                preconditions=tc.get("preconditions"),
                test_steps=json.dumps(steps, ensure_ascii=False),
                test_data=tc.get("test_data"),
                expected_result=tc.get("expected_result"),
                priority=tc.get("priority"),
                test_type=tc.get("test_type"),
                domain=d,
                comments=tc.get("comments"),
                source_story=story,
            )
            db.add(orm)
            db.flush()
            out.append(TestCaseOut(
                id=orm.id,
                external_id=orm.external_id,
                test_title=orm.test_title,
                description=orm.description,
                preconditions=orm.preconditions,
                test_steps=steps,
                test_data=orm.test_data,
                expected_result=orm.expected_result,
                priority=orm.priority,
                test_type=orm.test_type,
                domain=orm.domain,
                comments=orm.comments,
                source_story=orm.source_story,
            ))
    db.commit()
    return out


@app.get("/test-cases", response_model=List[TestCaseOut])
def list_test_cases(limit: int = 50, offset: int = 0, domain: Optional[str] = None, q: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(TestCaseORM)
    if domain:
        query = query.filter(TestCaseORM.domain == domain)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(TestCaseORM.test_title.ilike(like), TestCaseORM.description.ilike(like)))
    items = query.order_by(TestCaseORM.id.desc()).offset(offset).limit(limit).all()
    results: List[TestCaseOut] = []
    for it in items:
        steps = []
        try:
            steps = json.loads(it.test_steps) if it.test_steps else []
        except Exception:
            steps = []
        results.append(TestCaseOut(
            id=it.id,
            external_id=it.external_id,
            test_title=it.test_title,
            description=it.description,
            preconditions=it.preconditions,
            test_steps=steps,
            test_data=it.test_data,
            expected_result=it.expected_result,
            priority=it.priority,
            test_type=it.test_type,
            domain=it.domain,
            comments=it.comments,
            source_story=it.source_story,
        ))
    return results


@app.get("/test-cases/{item_id}", response_model=TestCaseOut)
def get_test_case(item_id: int, db: Session = Depends(get_db)):
    it = db.get(TestCaseORM, item_id)
    if not it:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        steps = json.loads(it.test_steps) if it.test_steps else []
    except Exception:
        steps = []
    return TestCaseOut(
        id=it.id,
        external_id=it.external_id,
        test_title=it.test_title,
        description=it.description,
        preconditions=it.preconditions,
        test_steps=steps,
        test_data=it.test_data,
        expected_result=it.expected_result,
        priority=it.priority,
        test_type=it.test_type,
        domain=it.domain,
        comments=it.comments,
        source_story=it.source_story,
    )


@app.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    items = db.query(TestCaseORM).order_by(TestCaseORM.id.asc()).all()
    def rowify(tc: TestCaseORM):
        try:
            steps = json.loads(tc.test_steps) if tc.test_steps else []
        except Exception:
            steps = []
        return [
            tc.id,
            tc.external_id or "",
            tc.test_title or "",
            tc.description or "",
            tc.preconditions or "",
            " | ".join(steps),
            tc.test_data or "",
            tc.expected_result or "",
            tc.priority or "",
            tc.test_type or "",
            tc.domain or "",
            tc.comments or "",
        ]
    headers = ["id", "external_id", "test_title", "description", "preconditions", "test_steps", "test_data", "expected_result", "priority", "test_type", "domain", "comments"]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    for it in items:
        writer.writerow(rowify(it))
    stream.seek(0)
    return StreamingResponse(iter([stream.read()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=test_cases.csv"})


@app.get("/export/excel")
def export_excel(db: Session = Depends(get_db)):
    items = db.query(TestCaseORM).order_by(TestCaseORM.id.asc()).all()
    rows = []
    for it in items:
        try:
            steps = json.loads(it.test_steps) if it.test_steps else []
        except Exception:
            steps = []
        rows.append({
            "id": it.id,
            "external_id": it.external_id,
            "test_title": it.test_title,
            "description": it.description,
            "preconditions": it.preconditions,
            "test_steps": " | ".join(steps),
            "test_data": it.test_data,
            "expected_result": it.expected_result,
            "priority": it.priority,
            "test_type": it.test_type,
            "domain": it.domain,
            "comments": it.comments,
        })
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Test Cases")
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=test_cases.xlsx"})


# Health
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(func.count(TestCaseORM.id)).scalar() or 0
    by_priority = dict((k or "", v) for k, v in db.query(TestCaseORM.priority, func.count(TestCaseORM.id)).group_by(TestCaseORM.priority).all())
    by_type = dict((k or "", v) for k, v in db.query(TestCaseORM.test_type, func.count(TestCaseORM.id)).group_by(TestCaseORM.test_type).all())
    by_domain = dict((k or "", v) for k, v in db.query(TestCaseORM.domain, func.count(TestCaseORM.id)).group_by(TestCaseORM.domain).all())
    return {
        "total": int(total),
        "by_priority": {k: int(v) for k, v in by_priority.items()},
        "by_type": {k: int(v) for k, v in by_type.items()},
        "by_domain": {k: int(v) for k, v in by_domain.items()},
    }
