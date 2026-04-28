"""Upload API: handle file uploads, extraction, and project creation."""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

router = APIRouter()

# In-memory store for uploaded projects (replace with DB in production)
_projects: dict[str, dict[str, Any]] = {}
UPLOAD_BASE = Path(tempfile.gettempdir()) / "sec-rag-uploads"
UPLOAD_BASE.mkdir(exist_ok=True)


class UploadResponse(BaseModel):
    project_id: str
    file_count: int
    files: list[str]


@router.post("/files", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    project_name: str = Form(""),
) -> UploadResponse:
    """Upload one or more files/folders (as zip) and create a project."""
    project_id = str(uuid4())[:8]
    project_dir = UPLOAD_BASE / project_id
    project_dir.mkdir(exist_ok=True)

    uploaded_files: list[str] = []

    for upload in files:
        dest_path = project_dir / (upload.filename or "unnamed")
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        # Auto-extract archives
        if dest_path.suffix == ".zip":
            extracted = _extract_zip(dest_path, project_dir)
            uploaded_files.extend(extracted)
            dest_path.unlink()  # Remove archive after extraction
        elif dest_path.suffix in {".tar", ".gz", ".tgz", ".bz2"}:
            extracted = _extract_tar(dest_path, project_dir)
            uploaded_files.extend(extracted)
            dest_path.unlink()
        else:
            uploaded_files.append(str(dest_path.relative_to(project_dir)))

    # Deduplicate and sort
    uploaded_files = sorted(set(uploaded_files))

    _projects[project_id] = {
        "id": project_id,
        "name": project_name or f"project-{project_id}",
        "dir": str(project_dir),
        "files": uploaded_files,
        "indexed": False,
    }

    logger.info(
        "Project {} created with {} files", project_id, len(uploaded_files)
    )
    return UploadResponse(
        project_id=project_id,
        file_count=len(uploaded_files),
        files=uploaded_files[:50],  # Limit for response
    )


@router.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    return list(_projects.values())


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return _projects[project_id]


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, str]:
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = Path(_projects[project_id]["dir"])
    if project_dir.exists():
        shutil.rmtree(project_dir)
    del _projects[project_id]
    return {"status": "deleted", "project_id": project_id}


def get_project_dir(project_id: str) -> Path:
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return Path(_projects[project_id]["dir"])


def mark_project_indexed(project_id: str) -> None:
    if project_id in _projects:
        _projects[project_id]["indexed"] = True


def is_project_indexed(project_id: str) -> bool:
    return _projects.get(project_id, {}).get("indexed", False)


def _extract_zip(archive: Path, dest: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.namelist():
            # Skip macOS metadata and hidden files
            if member.startswith("__MACOSX") or "/." in member:
                continue
            zf.extract(member, dest)
            member_path = dest / member
            if member_path.is_file():
                extracted.append(str(member_path.relative_to(dest)))
    return extracted


def _extract_tar(archive: Path, dest: Path) -> list[str]:
    extracted: list[str] = []
    mode = "r:gz" if archive.suffix in {".gz", ".tgz"} else "r:bz2" if archive.suffix == ".bz2" else "r"
    with tarfile.open(archive, mode) as tf:
        for member in tf.getmembers():
            if member.name.startswith(".") or "/." in member.name:
                continue
            tf.extract(member, dest)
            member_path = dest / member.name
            if member_path.is_file():
                extracted.append(str(member_path.relative_to(dest)))
    return extracted
