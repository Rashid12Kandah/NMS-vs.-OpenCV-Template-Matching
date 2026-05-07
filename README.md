# Digit Detection via Template Matching & Non-Maximum Suppression

> Locate every instance of a target digit ("5") in a multi-digit image, refine the candidate bounding boxes with a hand-rolled Non-Maximum Suppression routine, and study how the overlap threshold shapes the final result set.

## Overview

A compact computer-vision exercise built around OpenCV's `cv2.matchTemplate`. The workflow:

1. Load the source image and the template ("5").
2. Run normalized cross-correlation template matching against the source.
3. Threshold the response map to produce candidate boxes.
4. Apply Non-Maximum Suppression (NMS) to collapse near-duplicate boxes around each true detection.
5. Vary the overlap threshold to observe its effect on the surviving boxes.

## Tech Stack

| Library | Role |
|---|---|
| OpenCV (`cv2`) | Image I/O, template matching, bounding-box drawing |
| NumPy | Coordinate arrays, sorting, vectorized box math |
| Matplotlib | Inline visualization inside the notebook |

## Repository Structure

```
template-matching-nms-digits/
├── template_matching_nms.ipynb   # Main notebook
├── numbers.png                   # Source image (mixed digits)
├── numbers_template.png          # Template ("5")
└── README.md
```

## Quickstart

```bash
pip install opencv-python numpy matplotlib
jupyter notebook template_matching_nms.ipynb
```

The notebook expects `numbers.png` and `numbers_template.png` in the working directory.

## Methodology

### Part A — Template Matching with a Confidence Threshold

Normalized cross-correlation between the source and the template produces a response map. Pixels above the chosen threshold are treated as candidate top-left corners for matches.

```python
result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

threshold = 0.99
loc = np.where(result >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt,
                  (pt[0] + template.shape[1], pt[1] + template.shape[0]),
                  (0, 255, 0), 2)
```

`TM_CCOEFF_NORMED` was chosen for its robustness to brightness shifts, and the threshold of `0.99` keeps only very-high-confidence matches.

### Part B — Applying NMS to the Candidate Boxes

The thresholded candidate set typically contains multiple overlapping boxes around each true match. NMS keeps the strongest box per cluster and discards the rest.

```python
filtered_boxes = non_max_suppression(loc, overlap_thresh=0.3)

for box in filtered_boxes:
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
```

### Part C — Custom NMS Implementation

Standard greedy NMS: sort candidates by `y2`, repeatedly pick the bottom-most box, then suppress every remaining box whose overlap ratio with the picked one exceeds `overlap_thresh`.

```python
def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs,
            np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick]
```

The function expects each box in `[x1, y1, x2, y2]` form and returns the surviving boxes.

### Part D — Threshold Sensitivity

The `overlap_thresh` parameter governs how aggressively duplicates are removed: lower values suppress more boxes, higher values keep more. The notebook demonstrates the call at `overlap_thresh = 0.3`; sweeping across a range such as `[0.1, 0.3, 0.5, 0.7]` is the natural extension for characterising sensitivity.

## Results

After thresholded matching, each true "5" in `numbers.png` is surrounded by several overlapping rectangles. After NMS, those clusters collapse to a single bounding box per detection.

Place the rendered notebook output as `results/output.png` and reference it here once available.

## Engineering Considerations

- **Box construction for NMS.** The NMS function operates on `[x1, y1, x2, y2]` rectangles. To feed it the output of `cv2.matchTemplate` directly, candidate boxes should first be assembled from the `np.where` indices and the template's `(width, height)`, e.g.:
  ```python
  tH, tW = template_gray.shape
  boxes = [(x, y, x + tW, y + tH) for (x, y) in zip(*loc[::-1])]
  filtered_boxes = non_max_suppression(boxes, overlap_thresh=0.3)
  ```
- **Shared image canvas.** The notebook draws bounding boxes onto a single `image` array across cells, so the Part D visualization carries over rectangles drawn during Part A. Re-loading `image` before each visualization yields a cleaner per-stage comparison.
- **Threshold sweep.** A side-by-side panel of NMS outputs at multiple `overlap_thresh` values communicates Part D's findings more clearly than a single threshold.
- **Unused imports.** `skimage.metrics.ssim`, `PIL.Image`, `time`, and `imutils` are imported in the setup cell but not used downstream — safe to remove.

## License

This work is shared for educational use only. All rights reserved.

## Author

**Rashid Kandah**
