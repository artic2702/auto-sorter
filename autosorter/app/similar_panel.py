"""
"Find Similar" panel — shows the most similar faces to a selected face.

=== WHAT IS THIS FILE? ===
When the user clicks a face bounding box in the PhotoViewer, this panel
opens and shows the top-10 most similar faces ranked by cosine similarity.

This is the VISUAL PROOF that embeddings work:
  - Same person's faces should cluster at the top (score > 0.5)
  - Different people should have low scores (< 0.3)

=== HOW THE SEARCH WORKS ===
1. Load the "query" face's embedding from the database
2. Load ALL embeddings from the database into a numpy matrix
3. Compute dot product of query vs every other embedding (one operation!)
4. Sort by similarity, take top 10
5. Display as a grid with score labels

=== WHY IS THIS FAST? ===
Matrix multiplication! Instead of a Python loop comparing 100,000 pairs:
  similarities = all_embeddings @ query_embedding
This single numpy operation computes ALL similarities at once using
optimized C code. Even with 100k faces, it takes < 50ms.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


# How many similar faces to show
TOP_K = 10

# Similarity thresholds for color coding
THRESHOLD_HIGH = 0.50    # Green: very likely same person
THRESHOLD_MEDIUM = 0.30  # Yellow: possibly same person
# Below 0.30: Red — different person


class SimilarPanel(QWidget):
    """
    A panel that shows faces similar to a query face.

    Usage:
        panel = SimilarPanel(data_folder, query_face_id)
        panel.show()
    """

    def __init__(
        self,
        data_folder: Path,
        query_face_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._data_folder = data_folder
        self._query_face_id = query_face_id

        self.setWindowTitle("Find Similar Faces")
        self.resize(700, 500)
        self.setMinimumSize(400, 300)

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel layout and run the similarity search."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Open database and load the query face
        db = Database(self._data_folder / "library.db")
        query_face = db.get_face_by_id(self._query_face_id)

        if query_face is None or query_face["embedding"] is None:
            layout.addWidget(QLabel("❌ Face not found or has no embedding."))
            db.close()
            return

        # --- Query face header ---
        header = QHBoxLayout()

        query_thumb = QLabel()
        query_thumb.setFixedSize(112, 112)
        if query_face["thumb_path"]:
            pixmap = QPixmap(query_face["thumb_path"])
            if not pixmap.isNull():
                query_thumb.setPixmap(pixmap.scaled(
                    112, 112,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        query_thumb.setStyleSheet(
            "border: 3px solid #4CAF50; border-radius: 4px;"
        )
        header.addWidget(query_thumb)

        info = QVBoxLayout()
        info.addWidget(QLabel(
            "<b style='font-size: 16px;'>Query Face</b>"
        ))
        info.addWidget(QLabel(
            f"Confidence: {query_face['det_score']:.0%}"
        ))
        info.addWidget(QLabel(
            f"Quality: {query_face['quality']:.2f}" if query_face['quality'] else "Quality: N/A"
        ))
        info.addStretch()
        header.addLayout(info)
        header.addStretch()
        layout.addLayout(header)

        # --- Similarity results ---
        layout.addWidget(QLabel(
            "<b>Most Similar Faces</b> "
            "<span style='color: gray;'>(ranked by cosine similarity)</span>"
        ))

        # Run the search!
        query_embedding = FaceRecognizer.bytes_to_embedding(
            query_face["embedding"]
        )
        all_faces = db.get_all_embeddings()
        db.close()

        if len(all_faces) < 2:
            layout.addWidget(QLabel("Not enough faces to compare."))
            return

        # Build the embedding matrix and compute similarities
        results = self._find_similar(query_embedding, all_faces)

        # --- Results grid (in a scroll area) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(10)

        cols = 5  # 5 faces per row
        for i, (face_id, similarity, thumb_path, img_id) in enumerate(results):
            cell = self._make_face_cell(face_id, similarity, thumb_path)
            row = i // cols
            col = i % cols
            grid.addWidget(cell, row, col)

        scroll.setWidget(grid_container)
        layout.addWidget(scroll, stretch=1)

    def _find_similar(
        self,
        query_emb: np.ndarray,
        all_faces: list,
    ) -> list:
        """
        Find the top-K most similar faces to the query.

        HOW THIS WORKS:
        1. Stack all embeddings into a (N, 128) matrix
        2. Compute similarities = matrix @ query  → (N,) vector
        3. Sort descending, take top K (excluding the query itself)

        The matrix multiplication is the key insight:
          [face_1_emb]       [query]      [sim_1]
          [face_2_emb]   @   [  .  ]  =   [sim_2]
          [face_3_emb]       [  .  ]      [sim_3]
          [   ...     ]                    [ ... ]

        One operation, all similarities computed at once!
        """
        # Build arrays
        face_ids = []
        image_ids = []
        thumb_paths = []
        embeddings = []

        for face_row in all_faces:
            fid = face_row["id"]
            if fid == self._query_face_id:
                continue  # Skip the query face itself

            emb_blob = face_row["embedding"]
            if emb_blob is None:
                continue

            emb = FaceRecognizer.bytes_to_embedding(emb_blob)
            face_ids.append(fid)
            image_ids.append(face_row["image_id"])
            thumb_paths.append(face_row["thumb_path"])
            embeddings.append(emb)

        if not embeddings:
            return []

        # Stack into a (N, 128) matrix
        emb_matrix = np.stack(embeddings)

        # Compute all similarities at once: (N, 128) @ (128,) → (N,)
        similarities = emb_matrix @ query_emb

        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]

        # Take top K
        results = []
        for idx in sorted_indices[:TOP_K]:
            results.append((
                face_ids[idx],
                float(similarities[idx]),
                thumb_paths[idx],
                image_ids[idx],
            ))

        return results

    def _make_face_cell(
        self,
        face_id: int,
        similarity: float,
        thumb_path: str,
    ) -> QWidget:
        """Create a single face cell with thumbnail and score."""
        cell = QWidget()
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(4, 4, 4, 4)
        cell_layout.setSpacing(4)

        # Face thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(100, 100)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if thumb_path:
            pixmap = QPixmap(thumb_path)
            if not pixmap.isNull():
                thumb_label.setPixmap(pixmap.scaled(
                    96, 96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))

        # Color-code the border based on similarity
        if similarity >= THRESHOLD_HIGH:
            border_color = "#4CAF50"   # green — likely same person
        elif similarity >= THRESHOLD_MEDIUM:
            border_color = "#FFC107"   # yellow — uncertain
        else:
            border_color = "#F44336"   # red — different person

        thumb_label.setStyleSheet(
            f"border: 3px solid {border_color}; border-radius: 4px;"
        )
        cell_layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Similarity score label
        score_label = QLabel(f"{similarity:.2f}")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {border_color};"
        )
        cell_layout.addWidget(score_label)

        # Face ID (small, gray)
        id_label = QLabel(f"#{face_id}")
        id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        id_label.setStyleSheet("color: gray; font-size: 11px;")
        cell_layout.addWidget(id_label)

        return cell
