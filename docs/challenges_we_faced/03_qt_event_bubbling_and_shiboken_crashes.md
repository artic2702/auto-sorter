# Challenge 03 — Qt Event Bubbling & Shiboken C++ Conversion Crashes

## 🚨 The Symptom
During Milestone 5 development:
1. **Clicks Were Ignored**: Double-clicking a face thumbnail in the person album view or clicking a person avatar refused to open the photo viewer or trigger card actions.
2. **Fatal C++ Crash**: When attempting to open the photo viewer, the application crashed with a low-level Shiboken C++ exception:
   ```text
   Shiboken::Conversions::_pythonToCppCopy: Cannot copy-convert 000001C2B74600A0 (sqlite3.Row) to C++.
   Traceback (most recent call last):
     File "D:\Auto_sorter\autosorter\app\people_panel.py", line 523, in _on_open_full_photo
       "SELECT * FROM images WHERE id = ?", (face_row["image_id"],)
   KeyError: 'image_id'
   ```

---

## 🔍 Root Cause Analysis

### Issue 1: Mouse Event Interception by Child `QLabel`
In PySide6 / Qt, every `QWidget` receives mouse events (`mousePressEvent`, `mouseDoubleClickEvent`).
- We built a composite container widget `FaceCellWidget` containing an image preview `QLabel` inside a `QVBoxLayout`.
- By default in Qt, `QLabel` accepts mouse press events. When the user clicked on the face photo, the inner `QLabel` consumed the event, preventing the event from bubbling up to the parent `FaceCellWidget.mouseDoubleClickEvent`.
- As a result, double-clicks on the photo were silently swallowed.

---

### Issue 2: Shiboken C++ Marshaling of `sqlite3.Row`
PySide6 uses **Shiboken** to bridge Python objects to the underlying C++ Qt library.
- We defined a Qt Signal as:
  ```python
  open_photo_requested = Signal(dict)
  ```
- When a click occurred, we passed `face_row` directly:
  ```python
  self.open_photo_requested.emit(self._face_row)
  ```
- However, `self._face_row` was an internal `sqlite3.Row` object (not a standard Python `dict`).
- When Shiboken tried to pass this object across the C++ Qt signal boundary, it checked if `sqlite3.Row` had a registered C++ type converter for `dict`. It did not!
- This caused an immediate `Shiboken::Conversions::_pythonToCppCopy` type conversion fault. When Python attempted fallback recovery, dictionary key lookups failed with `KeyError`.

---

## 🛠️ The Solution

### Fix 1: Transparent Event Attribute on Child Widgets
To allow clicks to pass straight through the decorative image label to the parent cell container:
```python
self._thumb_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
self._name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
self._count_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
```
- Setting `WA_TransparentForMouseEvents = True` tells the Qt windowing system to ignore the label during hit testing. Clicks pass directly to the parent widget, triggering `mousePressEvent` and `mouseDoubleClickEvent` seamlessly.

---

### Fix 2: Type-Safe Signal Payloads using Primitives
Instead of passing complex Python objects (`sqlite3.Row` or arbitrary dictionaries) across the Qt C++ bridge:
```python
# In FaceCellWidget:
open_photo_requested = Signal(int)  # Passes face_id as a standard integer!

def mouseDoubleClickEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
        self.open_photo_requested.emit(self._face["id"])
```

In the receiving slot:
```python
def _on_open_full_photo(self, face_id: int) -> None:
    db = Database(self._data_folder / "library.db")
    face = db.get_face_by_id(face_id)
    image = db.get_image_by_id(face["image_id"])
    db.close()
    ...
```
- Passing native primitive types (`int`, `str`, `float`) guarantees 100% C++ type compatibility in Shiboken with zero serialization overhead.
- Querying by ID on-demand also avoids stale dictionary data if the row was modified concurrently.

---

## 🧠 Key Takeaways
1. In composite Qt widgets, always set `label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)` on decorative child labels if the parent widget needs to capture mouse clicks.
2. In PySide6 / PyQt, **never emit custom Python database wrapper objects (like `sqlite3.Row`) through Qt Signals**. Emit native integer IDs (`Signal(int)`) and let the receiver fetch the record from the database.
