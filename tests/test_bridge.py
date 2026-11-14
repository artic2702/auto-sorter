"""
End-to-end integration tests for the pure-Python background engine (`autosorter.core.engine`)
and Electron NDJSON RPC sidecar (`autosorter.bridge`).

Compatible with both `python -m unittest tests/test_bridge.py` and `pytest`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from autosorter.bridge import BridgeServer
from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


def _write_photo(path: Path, color_bgr=(60, 130, 85), size=(480, 640)) -> None:
    img = np.full((size[0], size[1], 3), color_bgr, dtype=np.uint8)
    cv2.circle(img, (size[1] // 2, size[0] // 2), 80, (230, 235, 230), -1)
    cv2.imwrite(str(path), img)


def _write_doc_image(path: Path) -> None:
    """Create a synthetic white page with dense horizontal black text-like lines."""
    img = np.full((700, 520, 3), 250, dtype=np.uint8)
    for y in range(50, 650, 18):
        cv2.line(img, (40, y), (470, y), (20, 20, 20), 2)
        for x in range(50, 460, 14):
            cv2.rectangle(img, (x, y - 5), (x + 7, y + 3), (15, 15, 15), -1)
    cv2.imwrite(str(path), img)


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float32)
    return v / np.linalg.norm(v)


class TestBridgeAndEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.photo_dir = self.root / "Photos"
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.root / "Exported"

        _write_photo(self.photo_dir / "img_one.jpg", (40, 120, 70))
        _write_photo(self.photo_dir / "img_two.jpg", (90, 90, 150))
        _write_doc_image(self.photo_dir / "receipt_scan.png")

        self.events: list[tuple[str, dict]] = []
        self.server = BridgeServer()
        self.server.emit_event = lambda ev, data: self.events.append((ev, data))  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _wait_until_idle(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.server.dispatch("pipeline.status", {})
            if not st["pipeline"]["running"] and not st["export"]["running"]:
                return
            time.sleep(0.05)
        self.fail("Timed out waiting for background runner to finish")

    def test_inspect_does_not_create_db_prematurely(self) -> None:
        res = self.server.dispatch("library.inspect", {"photo_folder": str(self.photo_dir)})
        self.assertTrue(res["folder_exists"])
        self.assertFalse(res["has_db"])
        self.assertEqual(res["action"], "start")
        self.assertEqual(res["probe"]["count"], 3)
        self.assertFalse((self.photo_dir / ".autosorter" / "library.db").exists())

    def test_full_pipeline_people_faces_albums_similarity_and_export(self) -> None:
        # 1. Run 4-stage pipeline on folder
        self.server.dispatch(
            "pipeline.start",
            {"photo_folder": str(self.photo_dir), "speed_mode": "quiet"},
        )
        self._wait_until_idle()

        event_names = [e[0] for e in self.events]
        self.assertIn("pipeline.stage", event_names)
        self.assertIn("pipeline.finished", event_names)

        inspect = self.server.dispatch("library.inspect", {"photo_folder": str(self.photo_dir)})
        self.assertTrue(inspect["has_db"])
        self.assertEqual(inspect["action"], "rescan")
        self.assertEqual(inspect["state"]["processed"], 3)

        # 2. Seed deterministic faces & 128-D embeddings to exercise People, Similarity, Group & Unknown albums
        data_folder = Path(inspect["data_folder"])
        db = Database(data_folder / "library.db")
        try:
            photos = self.server.dispatch("photos.list", {"photo_folder": str(self.photo_dir)})["photos"]
            self.assertEqual(len(photos), 3)
            img1_id = photos[0]["id"]
            img2_id = photos[1]["id"]

            emb_alice_1 = _unit_vec(101)
            emb_alice_2 = emb_alice_1 * 0.96 + _unit_vec(102) * 0.04
            emb_alice_2 = (emb_alice_2 / np.linalg.norm(emb_alice_2)).astype(np.float32)
            emb_bob = _unit_vec(202)
            emb_unknown = _unit_vec(303)

            # Clear any sentinel zero-face markers on img1 and img2
            db._conn.execute("DELETE FROM faces WHERE image_id IN (?, ?)", (img1_id, img2_id))
            f1 = db.insert_face(img1_id, 40, 40, 100, 100, 0.96, quality=0.88)
            f2 = db.insert_face(img2_id, 50, 50, 105, 105, 0.94, quality=0.85)
            f3 = db.insert_face(img2_id, 210, 55, 95, 95, 0.91, quality=0.80)
            f4 = db.insert_face(img1_id, 240, 60, 90, 90, 0.89, quality=0.76)

            db.update_embedding(f1, FaceRecognizer.embedding_to_bytes(emb_alice_1))
            db.update_embedding(f2, FaceRecognizer.embedding_to_bytes(emb_alice_2))
            db.update_embedding(f3, FaceRecognizer.embedding_to_bytes(emb_bob))
            db.update_embedding(f4, FaceRecognizer.embedding_to_bytes(emb_unknown))
            db._conn.commit()
        finally:
            db.close()

        # 3. Create Alice & Bob via RPC, leave f4 unassigned (Unknown Faces)
        alice = self.server.dispatch(
            "faces.create_person",
            {"photo_folder": str(self.photo_dir), "face_ids": [f1, f2], "display_name": "Alice"},
        )
        bob = self.server.dispatch(
            "faces.create_person",
            {"photo_folder": str(self.photo_dir), "face_ids": [f3], "display_name": "Bob"},
        )
        self.assertEqual(alice["display_name"], "Alice")
        self.assertEqual(bob["display_name"], "Bob")

        # 4. Verify People List & Detail
        plist = self.server.dispatch("people.list", {"photo_folder": str(self.photo_dir)})["persons"]
        self.assertEqual(len(plist), 2)
        alice_detail = self.server.dispatch(
            "people.get",
            {"photo_folder": str(self.photo_dir), "person_id": alice["person_id"]},
        )
        self.assertEqual(alice_detail["person"]["photo_count"], 2)
        self.assertEqual(len(alice_detail["faces"]), 2)

        # 5. Verify Cosine Similarity Search (`faces.similar`)
        sim = self.server.dispatch(
            "faces.similar",
            {"photo_folder": str(self.photo_dir), "face_id": f1, "top_k": 5},
        )
        self.assertEqual(sim["matches"][0]["face_id"], f2)
        self.assertGreater(sim["matches"][0]["similarity"], 0.9)

        # 6. Verify Smart Albums (img2 has Alice + Bob -> Group Photo; f4 -> Unknown Face)
        albums = {
            a["id"]: a
            for a in self.server.dispatch("albums.summary", {"photo_folder": str(self.photo_dir)})["albums"]
        }
        self.assertEqual(albums["group"]["count"], 1)
        self.assertEqual(albums["unknown"]["count"], 1)

        unknown_items = self.server.dispatch(
            "albums.items",
            {"photo_folder": str(self.photo_dir), "album_type": "unknown"},
        )
        self.assertTrue(unknown_items["is_face_album"])
        self.assertEqual(unknown_items["items"][0]["face_id"], f4)

        # 7. Rename, Set Cover, Reassign & Unassign
        self.server.dispatch(
            "people.rename",
            {"photo_folder": str(self.photo_dir), "person_id": bob["person_id"], "new_name": "Robert"},
        )
        self.server.dispatch(
            "people.set_cover",
            {"photo_folder": str(self.photo_dir), "person_id": alice["person_id"], "face_id": f2},
        )
        self.server.dispatch(
            "faces.reassign",
            {"photo_folder": str(self.photo_dir), "face_ids": [f4], "target_person_id": bob["person_id"]},
        )

        # 8. Safe Copy-Only Export
        source_before = sorted(p.name for p in self.photo_dir.iterdir() if p.is_file())
        self.server.dispatch(
            "export.start",
            {
                "photo_folder": str(self.photo_dir),
                "export_root": str(self.export_dir),
                "include_group": True,
                "include_docs": True,
                "include_unknown": False,
            },
        )
        self._wait_until_idle()

        self.assertTrue((self.export_dir / "Alice").is_dir())
        self.assertTrue((self.export_dir / "Robert").is_dir())
        self.assertTrue((self.export_dir / "Group Photos").is_dir())
        source_after = sorted(p.name for p in self.photo_dir.iterdir() if p.is_file())
        self.assertEqual(source_before, source_after)

        # 9. Merge Robert into Alice & Delete
        self.server.dispatch(
            "people.merge",
            {
                "photo_folder": str(self.photo_dir),
                "source_person_id": bob["person_id"],
                "target_person_id": alice["person_id"],
            },
        )
        remaining = self.server.dispatch("people.list", {"photo_folder": str(self.photo_dir)})["persons"]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], alice["person_id"])

    def test_tiff_preview_conversion_and_ndjson_subprocess(self) -> None:
        tiff_path = self.photo_dir / "sample_scan.tiff"
        img = np.full((300, 400, 3), (120, 180, 90), dtype=np.uint8)
        cv2.imwrite(str(tiff_path), img)

        preview = self.server.dispatch("image.preview", {"file_path": str(tiff_path), "max_dim": 800})
        self.assertTrue(Path(preview["preview_path"]).is_file())

        # Subprocess stdio smoke test
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "autosorter.bridge"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            assert proc.stdin and proc.stdout
            ready_line = proc.stdout.readline()
            self.assertEqual(json.loads(ready_line)["event"], "bridge.ready")

            proc.stdin.write(json.dumps({"id": 7, "method": "system.info", "params": {}}) + "\n")
            proc.stdin.write(json.dumps({"id": 8, "method": "bridge.shutdown", "params": {}}) + "\n")
            proc.stdin.flush()

            resp_info = json.loads(proc.stdout.readline())
            self.assertTrue(resp_info["ok"])
            self.assertEqual(resp_info["id"], 7)
            self.assertTrue(resp_info["result"]["models"]["yunet"]["present"])

            resp_bye = json.loads(proc.stdout.readline())
            self.assertEqual(resp_bye["id"], 8)
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
