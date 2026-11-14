"""
Headless NDJSON RPC bridge between Electron's main process and Auto-Sorter's core engine.

Protocol (UTF-8 newline-delimited JSON over stdin/stdout):
  Request:  {"id": 1, "method": "library.inspect", "params": {...}}
  Response: {"id": 1, "ok": true, "result": {...}}
            {"id": 1, "ok": false, "error": {"code": "...", "message": "..."}}
  Event:    {"event": "pipeline.progress", "data": {...}}

All Python logging is routed to stderr (and `.autosorter/autosorter.log`) so stdout
remains strictly machine-readable JSON lines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from autosorter import __version__
from autosorter.core.cluster import clean_up_person_album, recluster_all_auto_faces
from autosorter.core.db import Database
from autosorter.core.engine import ExportRunner, PipelineRunner
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.perf import (
    SpeedMode,
    get_cpu_count,
    get_rss_bytes,
    worker_count_for_mode,
)
from autosorter.core.scanner import SKIP_DIR_NAMES, is_image_file
from autosorter.core.thumbnailer import _read_image_safe
from autosorter.paths import SFACE_MODEL, YUNET_MODEL

log = logging.getLogger("autosorter.bridge")


class BridgeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BridgeServer:
    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._pipeline_runner: Optional[PipelineRunner] = None
        self._export_runner: Optional[ExportRunner] = None
        self._preview_cache_dir = Path(tempfile.gettempdir()) / "autosorter-electron-previews"

    # ------------------------------------------------------------------ I/O

    def _send(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def emit_event(self, event: str, data: Dict[str, Any]) -> None:
        self._send({"event": event, "data": data})

    # ------------------------------------------------------------------ DB helpers

    @staticmethod
    def _resolve_folders(params: Dict[str, Any]) -> tuple[Optional[Path], Path]:
        photo_raw = params.get("photo_folder")
        data_raw = params.get("data_folder")
        photo_folder = Path(photo_raw) if photo_raw else None
        if data_raw:
            data_folder = Path(data_raw)
        elif photo_folder is not None:
            data_folder = photo_folder / ".autosorter"
        else:
            raise BridgeError("BAD_REQUEST", "Either data_folder or photo_folder is required.")
        return photo_folder, data_folder

    @staticmethod
    def _open_db(data_folder: Path, *, must_exist: bool = True) -> Optional[Database]:
        db_path = data_folder / "library.db"
        if must_exist and not db_path.is_file():
            return None
        db = Database(db_path)
        db._conn.execute("PRAGMA busy_timeout = 5000")
        return db

    @staticmethod
    def _quick_folder_probe(root: Path, budget_sec: float = 0.18, cap: int = 2500) -> Dict[str, Any]:
        """Fast bounded file count preview before any DB exists."""
        if not root.is_dir():
            return {"count": 0, "capped": False}
        deadline = time.monotonic() + budget_sec
        stack = [str(root)]
        count = 0
        capped = False
        while stack:
            if time.monotonic() > deadline or count >= cap:
                capped = True
                break
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                                continue
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and is_image_file(entry.name):
                            count += 1
                            if count >= cap:
                                capped = True
                                break
            except OSError:
                continue
        return {"count": count, "capped": capped}

    # ------------------------------------------------------------------ RPC dispatch

    def dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        handlers = {
            "system.info": self._system_info,
            "library.inspect": self._library_inspect,
            "pipeline.start": self._pipeline_start,
            "pipeline.cancel": self._pipeline_cancel,
            "pipeline.status": self._pipeline_status,
            "people.list": self._people_list,
            "people.get": self._people_get,
            "people.rename": self._people_rename,
            "people.merge": self._people_merge,
            "people.delete": self._people_delete,
            "people.recheck": self._people_recheck,
            "people.set_cover": self._people_set_cover,
            "people.recluster": self._people_recluster,
            "faces.reassign": self._faces_reassign,
            "faces.create_person": self._faces_create_person,
            "faces.unassign": self._faces_unassign,
            "faces.similar": self._faces_similar,
            "photos.list": self._photos_list,
            "photos.get": self._photos_get,
            "albums.summary": self._albums_summary,
            "albums.items": self._albums_items,
            "export.start": self._export_start,
            "export.cancel": self._export_cancel,
            "image.preview": self._image_preview,
        }
        fn = handlers.get(method)
        if fn is None:
            raise BridgeError("UNKNOWN_METHOD", f"Unknown RPC method: {method}")
        return fn(params or {})

    # ------------------------------------------------------------------ System & Library

    def _system_info(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        rss = get_rss_bytes()
        return {
            "app_version": __version__,
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "cpu_count": get_cpu_count(),
            "speed_workers": {
                "quiet": worker_count_for_mode(SpeedMode.QUIET),
                "normal": worker_count_for_mode(SpeedMode.NORMAL),
                "fast": worker_count_for_mode(SpeedMode.FAST),
            },
            "rss_mb": round(rss / (1024 * 1024), 1) if rss > 0 else 0,
            "models": {
                "yunet": {
                    "present": YUNET_MODEL.is_file(),
                    "name": YUNET_MODEL.name,
                    "size_kb": round(YUNET_MODEL.stat().st_size / 1024, 1) if YUNET_MODEL.is_file() else 0,
                },
                "sface": {
                    "present": SFACE_MODEL.is_file(),
                    "name": SFACE_MODEL.name,
                    "size_mb": round(SFACE_MODEL.stat().st_size / (1024 * 1024), 1) if SFACE_MODEL.is_file() else 0,
                },
            },
        }

    def _library_inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        photo_folder, data_folder = self._resolve_folders(params)
        folder_exists = photo_folder.is_dir() if photo_folder else False

        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            probe = self._quick_folder_probe(photo_folder) if (photo_folder and folder_exists) else {"count": 0, "capped": False}
            return {
                "photo_folder": str(photo_folder) if photo_folder else None,
                "data_folder": str(data_folder),
                "folder_exists": folder_exists,
                "has_db": False,
                "action": "start",
                "probe": probe,
                "state": {
                    "total": 0,
                    "processed": 0,
                    "pending": 0,
                    "missing": 0,
                    "error": 0,
                    "faces": 0,
                    "embedded": 0,
                    "unassigned": 0,
                    "persons": 0,
                    "named_persons": 0,
                    "needs_detection": 0,
                },
                "albums": {"group": 0, "unknown": 0, "docs": 0},
            }

        try:
            state = db.get_pipeline_state()
            named_persons = db._conn.execute(
                "SELECT COUNT(*) FROM persons WHERE hidden = 0 AND merged_into IS NULL AND name_is_user_set = 1"
            ).fetchone()[0]
            state["named_persons"] = named_persons

            group_count = len(db.get_group_photos())
            unknown_count = len(db.get_unknown_faces())
            docs_count = len(db.get_text_and_docs())

            needs_embed = max(0, state["faces"] - state["embedded"])
            needs_initial_cluster = state["persons"] == 0 and state["embedded"] > 0
            interrupted = (
                state["pending"] > 0
                or state["needs_detection"] > 0
                or needs_embed > 0
                or needs_initial_cluster
            )
            if state["total"] == 0:
                action = "start"
            elif interrupted:
                action = "resume"
            else:
                action = "rescan"

            return {
                "photo_folder": str(photo_folder) if photo_folder else None,
                "data_folder": str(data_folder),
                "folder_exists": folder_exists,
                "has_db": True,
                "action": action,
                "probe": None,
                "state": state,
                "albums": {
                    "group": group_count,
                    "unknown": unknown_count,
                    "docs": docs_count,
                },
            }
        finally:
            db.close()

    # ------------------------------------------------------------------ Pipeline

    def _pipeline_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._pipeline_runner and self._pipeline_runner.is_running:
            raise BridgeError("ALREADY_RUNNING", "A scan is already running.")

        photo_folder, data_folder = self._resolve_folders(params)
        if photo_folder is None or not photo_folder.is_dir():
            raise BridgeError("FOLDER_NOT_FOUND", "Select an existing photo folder first.")

        mode_str = str(params.get("speed_mode") or "normal").lower()
        speed_map = {
            "quiet": SpeedMode.QUIET,
            "normal": SpeedMode.NORMAL,
            "fast": SpeedMode.FAST,
        }
        speed_mode = speed_map.get(mode_str, SpeedMode.NORMAL)

        def on_stage(stage_name: str, stage_num: int, total_stages: int) -> None:
            self.emit_event(
                "pipeline.stage",
                {
                    "stage_name": stage_name,
                    "stage_num": stage_num,
                    "total_stages": total_stages,
                },
            )

        def on_progress(done: int, total: int, detail: str, stage_num: int, stage_name: str) -> None:
            rss = get_rss_bytes()
            self.emit_event(
                "pipeline.progress",
                {
                    "done": done,
                    "total": total,
                    "detail": detail,
                    "stage_num": stage_num,
                    "stage_name": stage_name,
                    "rss_mb": round(rss / (1024 * 1024), 1) if rss > 0 else 0,
                },
            )

        def on_finished(summary: dict) -> None:
            self.emit_event("pipeline.finished", summary)

        def on_error(message: str) -> None:
            self.emit_event("pipeline.error", {"message": message})

        self._pipeline_runner = PipelineRunner(
            photo_folder=photo_folder,
            data_folder=data_folder,
            speed_mode=speed_mode,
            on_stage=on_stage,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
        )
        self._pipeline_runner.start()
        return self._pipeline_runner.snapshot

    def _pipeline_cancel(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        if self._pipeline_runner and self._pipeline_runner.is_running:
            self._pipeline_runner.cancel()
            return {"cancelling": True}
        return {"cancelling": False}

    def _pipeline_status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pipeline": self._pipeline_runner.snapshot if self._pipeline_runner else {"running": False},
            "export": self._export_runner.snapshot if self._export_runner else {"running": False},
        }

    # ------------------------------------------------------------------ People

    def _people_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            return {"persons": []}
        try:
            rows = db._conn.execute(
                """
                SELECT p.id,
                       p.display_name,
                       p.name_is_user_set,
                       p.cover_face_id,
                       p.face_count,
                       cf.thumb_path AS cover_thumb_path,
                       (SELECT COUNT(DISTINCT f.image_id)
                          FROM faces f
                          JOIN images i ON i.id = f.image_id
                         WHERE f.person_id = p.id
                           AND f.det_score > 0
                           AND i.status = 'processed') AS photo_count
                FROM persons p
                LEFT JOIN faces cf ON cf.id = p.cover_face_id
                WHERE p.hidden = 0 AND p.merged_into IS NULL
                ORDER BY p.face_count DESC, p.id ASC
                """
            ).fetchall()
            return {
                "persons": [
                    {
                        "id": r["id"],
                        "display_name": r["display_name"],
                        "name_is_user_set": bool(r["name_is_user_set"]),
                        "cover_face_id": r["cover_face_id"],
                        "cover_thumb_path": r["cover_thumb_path"],
                        "face_count": r["face_count"],
                        "photo_count": r["photo_count"],
                    }
                    for r in rows
                ]
            }
        finally:
            db.close()

    def _people_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        person_id = int(params["person_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Library database does not exist yet.")
        try:
            person = db._conn.execute(
                """
                SELECT p.id, p.display_name, p.name_is_user_set, p.cover_face_id, p.face_count,
                       cf.thumb_path AS cover_thumb_path
                FROM persons p
                LEFT JOIN faces cf ON cf.id = p.cover_face_id
                WHERE p.id = ? AND p.hidden = 0 AND p.merged_into IS NULL
                """,
                (person_id,),
            ).fetchone()
            if person is None:
                raise BridgeError("NOT_FOUND", "Person not found.")

            face_rows = db._conn.execute(
                """
                SELECT f.id, f.image_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                       f.det_score, f.quality, f.thumb_path, f.assign_source, f.assign_conf,
                       i.path AS image_path, i.thumb_path AS image_thumb,
                       i.width AS image_width, i.height AS image_height, i.taken_at
                FROM faces f
                JOIN images i ON i.id = f.image_id
                WHERE f.person_id = ? AND f.det_score > 0 AND i.status = 'processed'
                ORDER BY f.quality DESC, f.id ASC
                """,
                (person_id,),
            ).fetchall()

            faces = [
                {
                    "id": r["id"],
                    "image_id": r["image_id"],
                    "bbox": [r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]],
                    "det_score": round(float(r["det_score"] or 0), 4),
                    "quality": round(float(r["quality"] or 0), 4),
                    "thumb_path": r["thumb_path"],
                    "assign_source": r["assign_source"] or "auto",
                    "assign_conf": round(float(r["assign_conf"] or 0), 4) if r["assign_conf"] is not None else None,
                    "image_path": r["image_path"],
                    "filename": os.path.basename(r["image_path"] or ""),
                    "image_thumb": r["image_thumb"],
                    "image_width": r["image_width"],
                    "image_height": r["image_height"],
                    "taken_at": r["taken_at"],
                    "is_cover": r["id"] == person["cover_face_id"],
                }
                for r in face_rows
            ]
            photo_count = len({f["image_id"] for f in faces})
            return {
                "person": {
                    "id": person["id"],
                    "display_name": person["display_name"],
                    "name_is_user_set": bool(person["name_is_user_set"]),
                    "cover_face_id": person["cover_face_id"],
                    "cover_thumb_path": person["cover_thumb_path"],
                    "face_count": person["face_count"],
                    "photo_count": photo_count,
                },
                "faces": faces,
            }
        finally:
            db.close()

    def _people_rename(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        person_id = int(params["person_id"])
        new_name = str(params.get("new_name") or "").strip()
        if not new_name:
            raise BridgeError("VALIDATION_ERROR", "Name cannot be empty.")
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            db.rename_person(person_id, new_name)
            return {"id": person_id, "display_name": new_name, "name_is_user_set": True}
        finally:
            db.close()

    def _people_merge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        source_id = int(params["source_person_id"])
        target_id = int(params["target_person_id"])
        if source_id == target_id:
            raise BridgeError("VALIDATION_ERROR", "Cannot merge a person into themselves.")
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            db.merge_persons(source_person_id=source_id, target_person_id=target_id)
            return {"merged_into": target_id}
        finally:
            db.close()

    def _people_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        person_id = int(params["person_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            db.delete_person(person_id)
            return {"deleted": person_id}
        finally:
            db.close()

    def _people_recheck(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        person_id = int(params["person_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            moved = clean_up_person_album(db, person_id)
            total_shifted = sum(moved.values())
            return {"moved": moved, "total_shifted": total_shifted}
        finally:
            db.close()

    def _people_set_cover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        person_id = int(params["person_id"])
        face_id = int(params["face_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            db.update_person_cover(person_id, face_id)
            return {"person_id": person_id, "cover_face_id": face_id}
        finally:
            db.close()

    def _people_recluster(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._pipeline_runner and self._pipeline_runner.is_running:
            raise BridgeError("BUSY", "Wait for the current scan to finish before re-clustering.")
        _, data_folder = self._resolve_folders(params)
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            assigned, persons_count = recluster_all_auto_faces(db)
            return {"faces_assigned": assigned, "persons_count": persons_count}
        finally:
            db.close()

    # ------------------------------------------------------------------ Face Corrections & Similarity

    @staticmethod
    def _extract_face_ids(params: Dict[str, Any]) -> List[int]:
        raw_ids = params.get("face_ids")
        if raw_ids is not None:
            ids = [int(x) for x in raw_ids]
        elif "face_id" in params:
            ids = [int(params["face_id"])]
        else:
            ids = []
        if not ids:
            raise BridgeError("VALIDATION_ERROR", "At least one face_id is required.")
        return ids

    def _faces_reassign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        face_ids = self._extract_face_ids(params)
        target_person_id = int(params["target_person_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            for fid in face_ids:
                db.reassign_face(fid, target_person_id)
            return {"moved_count": len(face_ids), "target_person_id": target_person_id}
        finally:
            db.close()

    def _faces_create_person(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        face_ids = self._extract_face_ids(params)
        display_name = str(params.get("display_name") or "").strip() or None
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            person_id = db.create_person_for_face(face_ids[0], display_name=display_name)
            if display_name:
                db.rename_person(person_id, display_name)
            for fid in face_ids[1:]:
                db.reassign_face(fid, person_id)
            person = db.get_person_by_id(person_id)
            return {
                "person_id": person_id,
                "display_name": person["display_name"] if person else (display_name or f"Person {person_id}"),
                "moved_count": len(face_ids),
            }
        finally:
            db.close()

    def _faces_unassign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        face_ids = self._extract_face_ids(params)
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            for fid in face_ids:
                db.unassign_face(fid)
            return {"unassigned_count": len(face_ids)}
        finally:
            db.close()

    def _faces_similar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        face_id = int(params["face_id"])
        top_k = max(1, min(int(params.get("top_k") or 12), 48))

        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            query_row = db._conn.execute(
                """
                SELECT f.*, i.path AS image_path, p.display_name AS person_name
                FROM faces f
                JOIN images i ON i.id = f.image_id
                LEFT JOIN persons p ON p.id = f.person_id
                WHERE f.id = ?
                """,
                (face_id,),
            ).fetchone()
            if query_row is None or query_row["embedding"] is None:
                raise BridgeError("NOT_FOUND", "Selected face has no embedding.")

            query_emb = FaceRecognizer.bytes_to_embedding(query_row["embedding"])
            candidates = db._conn.execute(
                """
                SELECT f.id, f.image_id, f.embedding, f.thumb_path, f.quality, f.det_score,
                       f.person_id, p.display_name AS person_name, i.path AS image_path
                FROM faces f
                JOIN images i ON i.id = f.image_id
                LEFT JOIN persons p ON p.id = f.person_id
                WHERE f.embedding IS NOT NULL
                  AND f.det_score > 0
                  AND f.id != ?
                  AND i.status = 'processed'
                """,
                (face_id,),
            ).fetchall()

            query_info = {
                "id": query_row["id"],
                "image_id": query_row["image_id"],
                "image_path": query_row["image_path"],
                "filename": os.path.basename(query_row["image_path"] or ""),
                "thumb_path": query_row["thumb_path"],
                "det_score": round(float(query_row["det_score"] or 0), 4),
                "quality": round(float(query_row["quality"] or 0), 4),
                "person_id": query_row["person_id"],
                "person_name": query_row["person_name"],
            }

            if not candidates:
                return {"query": query_info, "matches": []}

            emb_matrix = np.stack(
                [FaceRecognizer.bytes_to_embedding(r["embedding"]) for r in candidates]
            )
            sims = emb_matrix @ query_emb
            top_indices = np.argsort(sims)[::-1][:top_k]

            matches = []
            for idx in top_indices:
                r = candidates[int(idx)]
                sim_val = float(sims[int(idx)])
                matches.append(
                    {
                        "face_id": r["id"],
                        "image_id": r["image_id"],
                        "image_path": r["image_path"],
                        "filename": os.path.basename(r["image_path"] or ""),
                        "thumb_path": r["thumb_path"],
                        "similarity": round(sim_val, 4),
                        "quality": round(float(r["quality"] or 0), 4),
                        "person_id": r["person_id"],
                        "person_name": r["person_name"],
                    }
                )
            return {"query": query_info, "matches": matches}
        finally:
            db.close()

    # ------------------------------------------------------------------ Photos

    def _photos_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            return {"photos": [], "total": 0}

        filter_mode = str(params.get("filter") or "all").lower()
        person_id = params.get("person_id")
        search = str(params.get("search") or "").strip().lower()

        try:
            where_clauses = ["i.status != 'missing'"]
            sql_params: List[Any] = []

            if filter_mode == "errors":
                where_clauses.append("i.status = 'error'")
            else:
                where_clauses.append("i.status = 'processed'")
                where_clauses.append("i.thumb_path IS NOT NULL")

            if person_id not in (None, "", 0, "0"):
                where_clauses.append(
                    "EXISTS (SELECT 1 FROM faces pf WHERE pf.image_id = i.id AND pf.person_id = ? AND pf.det_score > 0)"
                )
                sql_params.append(int(person_id))

            if filter_mode == "with_faces":
                where_clauses.append(
                    "EXISTS (SELECT 1 FROM faces wf WHERE wf.image_id = i.id AND wf.det_score > 0)"
                )
            elif filter_mode == "no_faces":
                where_clauses.append(
                    "NOT EXISTS (SELECT 1 FROM faces nf WHERE nf.image_id = i.id AND nf.det_score > 0)"
                )
            elif filter_mode == "docs":
                where_clauses.append("i.doc_score >= 0.5")

            if search:
                where_clauses.append("LOWER(i.path) LIKE ?")
                sql_params.append(f"%{search}%")

            sql = f"""
                SELECT i.id, i.path, i.size, i.mtime, i.width, i.height,
                       i.taken_at, i.thumb_path, i.doc_score, i.status, i.error,
                       (SELECT COUNT(*) FROM faces f WHERE f.image_id = i.id AND f.det_score > 0) AS face_count
                FROM images i
                WHERE {' AND '.join(where_clauses)}
                ORDER BY COALESCE(i.taken_at, '') DESC, i.id DESC
            """
            rows = db._conn.execute(sql, sql_params).fetchall()
            photos = [
                {
                    "id": r["id"],
                    "path": r["path"],
                    "filename": os.path.basename(r["path"] or ""),
                    "size": r["size"],
                    "mtime": r["mtime"],
                    "width": r["width"],
                    "height": r["height"],
                    "taken_at": r["taken_at"],
                    "thumb_path": r["thumb_path"],
                    "doc_score": round(float(r["doc_score"] or 0), 3),
                    "status": r["status"],
                    "error": r["error"],
                    "face_count": r["face_count"],
                }
                for r in rows
            ]
            return {"photos": photos, "total": len(photos)}
        finally:
            db.close()

    def _photos_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        image_id = int(params["image_id"])
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            raise BridgeError("NOT_FOUND", "Database not found.")
        try:
            img = db.get_image_by_id(image_id)
            if img is None:
                raise BridgeError("NOT_FOUND", "Photo not found.")

            face_rows = db._conn.execute(
                """
                SELECT f.*, p.display_name AS person_name
                FROM faces f
                LEFT JOIN persons p ON p.id = f.person_id
                WHERE f.image_id = ? AND f.det_score > 0
                ORDER BY f.bbox_x ASC, f.id ASC
                """,
                (image_id,),
            ).fetchall()

            faces = []
            for r in face_rows:
                landmarks = None
                if r["landmarks"]:
                    try:
                        landmarks = json.loads(r["landmarks"])
                    except (json.JSONDecodeError, TypeError):
                        landmarks = None
                faces.append(
                    {
                        "id": r["id"],
                        "image_id": r["image_id"],
                        "bbox": [r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]],
                        "landmarks": landmarks,
                        "det_score": round(float(r["det_score"] or 0), 4),
                        "quality": round(float(r["quality"] or 0), 4),
                        "thumb_path": r["thumb_path"],
                        "person_id": r["person_id"],
                        "person_name": r["person_name"],
                        "assign_source": r["assign_source"] or "auto",
                        "assign_conf": round(float(r["assign_conf"] or 0), 4) if r["assign_conf"] is not None else None,
                    }
                )

            return {
                "photo": {
                    "id": img["id"],
                    "path": img["path"],
                    "filename": os.path.basename(img["path"] or ""),
                    "size": img["size"],
                    "mtime": img["mtime"],
                    "width": img["width"],
                    "height": img["height"],
                    "taken_at": img["taken_at"],
                    "thumb_path": img["thumb_path"],
                    "doc_score": round(float(img["doc_score"] or 0), 3),
                    "status": img["status"],
                    "error": img["error"],
                },
                "faces": faces,
            }
        finally:
            db.close()

    # ------------------------------------------------------------------ Smart Albums & Export

    def _albums_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            return {
                "albums": [
                    {"id": "group", "title": "Group Photos", "count": 0, "previews": []},
                    {"id": "unknown", "title": "Unknown Faces", "count": 0, "previews": []},
                    {"id": "docs", "title": "Text & Documents", "count": 0, "previews": []},
                ]
            }
        try:
            groups = db.get_group_photos()
            unknowns = db.get_unknown_faces()
            docs = db.get_text_and_docs()
            return {
                "albums": [
                    {
                        "id": "group",
                        "title": "Group Photos",
                        "subtitle": "Photos where 2 or more distinct people appear together",
                        "unit": "photo",
                        "count": len(groups),
                        "previews": [r["thumb_path"] for r in groups[:4] if r["thumb_path"]],
                    },
                    {
                        "id": "unknown",
                        "title": "Unknown Faces",
                        "subtitle": "Detected faces awaiting manual assignment or new person creation",
                        "unit": "face",
                        "count": len(unknowns),
                        "previews": [r["thumb_path"] for r in unknowns[:4] if r["thumb_path"]],
                    },
                    {
                        "id": "docs",
                        "title": "Text & Documents",
                        "subtitle": "Screenshots, receipts, notes, and text-heavy captures",
                        "unit": "item",
                        "count": len(docs),
                        "previews": [r["thumb_path"] for r in docs[:4] if r["thumb_path"]],
                    },
                ]
            }
        finally:
            db.close()

    def _albums_items(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data_folder = self._resolve_folders(params)
        album_type = str(params.get("album_type") or "").lower()
        db = self._open_db(data_folder, must_exist=True)
        if db is None:
            return {"album_type": album_type, "items": []}
        try:
            if album_type == "group":
                rows = db.get_group_photos()
                items = [
                    {
                        "id": r["id"],
                        "image_id": r["id"],
                        "path": r["path"],
                        "filename": os.path.basename(r["path"] or ""),
                        "thumb_path": r["thumb_path"],
                        "width": r["width"],
                        "height": r["height"],
                        "taken_at": r["taken_at"],
                    }
                    for r in rows
                ]
                return {"album_type": "group", "is_face_album": False, "items": items}

            if album_type == "unknown":
                rows = db.get_unknown_faces()
                items = [
                    {
                        "id": r["id"],
                        "face_id": r["id"],
                        "image_id": r["image_id"],
                        "path": r["image_path"],
                        "filename": os.path.basename(r["image_path"] or ""),
                        "thumb_path": r["thumb_path"],
                        "image_thumb": r["image_thumb"],
                        "det_score": round(float(r["det_score"] or 0), 4),
                        "quality": round(float(r["quality"] or 0), 4),
                    }
                    for r in rows
                ]
                return {"album_type": "unknown", "is_face_album": True, "items": items}

            if album_type == "docs":
                rows = db.get_text_and_docs()
                items = [
                    {
                        "id": r["id"],
                        "image_id": r["id"],
                        "path": r["path"],
                        "filename": os.path.basename(r["path"] or ""),
                        "thumb_path": r["thumb_path"],
                        "width": r["width"],
                        "height": r["height"],
                        "doc_score": round(float(r["doc_score"] or 0), 3),
                        "taken_at": r["taken_at"],
                    }
                    for r in rows
                ]
                return {"album_type": "docs", "is_face_album": False, "items": items}

            raise BridgeError("VALIDATION_ERROR", f"Unsupported album_type: {album_type}")
        finally:
            db.close()

    def _export_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._export_runner and self._export_runner.is_running:
            raise BridgeError("ALREADY_RUNNING", "An export is already in progress.")

        _, data_folder = self._resolve_folders(params)
        export_raw = params.get("export_root")
        if not export_raw:
            raise BridgeError("VALIDATION_ERROR", "Select an export destination folder.")
        export_root = Path(export_raw)

        db_path = data_folder / "library.db"
        if not db_path.is_file():
            raise BridgeError("NOT_FOUND", "Scan a photo library before exporting.")

        raw_pids = params.get("person_ids")
        person_ids = [int(x) for x in raw_pids] if raw_pids is not None else None

        self._export_runner = ExportRunner(
            data_folder=data_folder,
            export_root=export_root,
            person_ids=person_ids,
            include_group=bool(params.get("include_group", True)),
            include_unknown=bool(params.get("include_unknown", False)),
            include_docs=bool(params.get("include_docs", True)),
            on_progress=lambda done, total, detail: self.emit_event(
                "export.progress", {"done": done, "total": total, "detail": detail}
            ),
            on_finished=lambda result: self.emit_event("export.finished", result),
            on_error=lambda message: self.emit_event("export.error", {"message": message}),
        )
        self._export_runner.start()
        return self._export_runner.snapshot

    def _export_cancel(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        if self._export_runner and self._export_runner.is_running:
            self._export_runner.cancel()
            return {"cancelling": True}
        return {"cancelling": False}

    # ------------------------------------------------------------------ Non-browser format preview (HEIC / RAW / TIFF)

    def _image_preview(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_path = str(params.get("file_path") or "")
        max_dim = max(256, min(int(params.get("max_dim") or 2200), 4096))
        if not file_path or not os.path.isfile(file_path):
            raise BridgeError("NOT_FOUND", f"File not found: {file_path}")

        st = os.stat(file_path)
        cache_key = hashlib.sha1(
            f"{file_path}|{st.st_size}|{st.st_mtime}|{max_dim}".encode("utf-8", errors="replace")
        ).hexdigest()
        self._preview_cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._preview_cache_dir / f"{cache_key}.jpg"

        if out_path.is_file() and out_path.stat().st_size > 0:
            return {"preview_path": str(out_path)}

        img = _read_image_safe(file_path, max_decode_dim=max_dim)
        if img is None:
            raise BridgeError("DECODE_ERROR", "Could not decode image file.")

        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise BridgeError("DECODE_ERROR", "Could not encode preview JPEG.")
        encoded.tofile(str(out_path))
        return {"preview_path": str(out_path)}

    # ------------------------------------------------------------------ Main loop

    def serve_forever(self) -> int:
        self.emit_event("bridge.ready", {"version": __version__, "pid": os.getpid()})
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            req_id = None
            try:
                msg = json.loads(line)
                req_id = msg.get("id")
                method = msg.get("method", "")
                if method == "bridge.shutdown":
                    if self._pipeline_runner and self._pipeline_runner.is_running:
                        self._pipeline_runner.cancel()
                    if self._export_runner and self._export_runner.is_running:
                        self._export_runner.cancel()
                    if req_id is not None:
                        self._send({"id": req_id, "ok": True, "result": {"bye": True}})
                    return 0

                result = self.dispatch(method, msg.get("params") or {})
                if req_id is not None:
                    self._send({"id": req_id, "ok": True, "result": result})
            except BridgeError as berr:
                if req_id is not None:
                    self._send(
                        {
                            "id": req_id,
                            "ok": False,
                            "error": {"code": berr.code, "message": berr.message},
                        }
                    )
            except Exception as exc:
                log.exception("Unhandled error in RPC bridge")
                if req_id is not None:
                    self._send(
                        {
                            "id": req_id,
                            "ok": False,
                            "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
                        }
                    )
        return 0


def main() -> int:
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = BridgeServer()
    return server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
