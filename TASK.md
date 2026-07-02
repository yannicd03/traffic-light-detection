# ML2 Bonus — Task Document

**Course:** Maschinelles Lernen II: Fortgeschrittene Verfahren (KIT / aifb), Sommersemester 2026
**Prof:** Dr. J.M. Zöllner · **Tutors:** Nikolai Polley, Marcus Fechner (nikolai.polley@kit.edu)
**Window:** 16.06.2026 → **31.07.2026, 23:55** · Submission via **Ilias** (Forum for questions)

---

## 0. Organisation (hard rules)

- Work in groups of **max. 3** (smaller / solo allowed). One submission per group, **not** three identical ones.
- Up to **3 bonus points** → exam grade improvement (0.3 / 0.4 steps). Only counts **if you pass the exam**. All group members get the same points.
- Two projects offered this semester; you may do both but **only one** grants bonus points. **Pick one** (see decision below).
- Bonus valid for exam in **SS2026 and WS2026/27** only.
- A "Teamführender" (team lead) must add the up-to-2 other members in Ilias via "Team verwalten" using their `u****` short ID **before the deadline** (no post-deadline team assignment possible). Lead uploads all docs.
- Group with **>3 members ⇒ not graded.** Do not add uninvolved students.
- **Do not redistribute the dataset.**
- All aids allowed: foreign public code, ChatGPT/Codex/Claude/Copilot, Stack Overflow, pretrained models (ImageNet etc.). Caveat: a 100% generated solution teaches you nothing. **No plagiarism** (anti-plagiarism software runs; 3-person group with 3 separate identical submissions = flagged as plagiarism).

---

## 1. Decision: which project?

> **Action required:** choose one. Recommendation below.

| | **Colorization** | **Traffic Light Detection (TLD)** |
|---|---|---|
| Task | Gray → RGB image generation | 2D object detection + classification of traffic lights |
| Output | `prediction.npy`, shape `[50,224,224,3]`, uint8 | `predictions.csv` (bbox + conf + class) |
| Metric | MSE vs. held-out RGB | F1 (IoU > 0.5 + class match) |
| Code volume | Higher (custom model/training loop) | Lower (off-the-shelf YOLO / RT-DETR) |
| Framework | PyTorch (U-Net / CNN / GAN / Diffusion) | Ultralytics YOLO / RT-DETR |

**Recommendation:** TLD is the lower-effort path to bonus points — the ATLAS dataset is already in YOLO format and Ultralytics trains a detector in a few lines. Colorization is the more interesting/educational self-supervised problem but needs a custom model + training loop.

---

## 2. Project A — Colorization

**Goal:** train a model that produces a colored RGB image from a black-and-white input.
Dataset: https://bwsyncandshare.kit.edu/s/KcigioLJHmaKD8n
Test inputs to colorize: `/student_dataset/test_color/images/` (**50 grayscale images**).

### Background / hints
- Grayscale is underdetermined: `grey_white = 0.2125·r + 0.7154·g + 0.0721·b` (3 unknowns, 1 equation) → many RGB combos map to same gray. A learned model still gives good estimates (e.g. grass is usually green).
- Self-supervised: no explicit labels needed — any RGB image can be a training example (label = original RGB, input = its grayscale via `skimage.color.rgb2gray`).
- **Strongly recommended: predict in Lab color space, not RGB.** Channel 1 = lightness (= the given gray image, use directly), channels 2 & 3 = color → model only predicts **2 channels** instead of 3. Convert back to RGB for submission (`skimage.color.rgb2lab` / `lab2rgb`).

### Tasks
- [ ] **Dataset + Dataloader:** read training RGB images, derive grayscale inputs (`rgb2gray`). (Optionally convert to Lab.)
- [ ] **Model:** input = gray, output = RGB (same H×W as input). Options:
  - U-Net or similar (recommended) — implement upsampling via `nn.Upsample` or `nn.ConvTranspose2d`. Can add attention blocks / transformer.
  - Self-built CNN.
  - Generative: GAN or diffusion conditioned on the gray image.
  - Framing: as semantic segmentation, RGB channels = classes → regression problem.
- [ ] **Train** (consider GPU / Colab — see §6).
- [ ] **Predict** the 50 test grayscale images, in original order (**do not reorder test images**).
- [ ] **Post-process to submission format:**
  - If trained in Lab/normalized space, convert predictions **back to RGB**.
  - PyTorch is channels-first `[50,3,224,224]` → transpose to `[50,224,224,3]`: `arr = arr.transpose((1,2,0))` per image (`[3,224,224]→[224,224,3]`).
  - If values in `[0,1]` float: `pred = (preds*255).astype(np.uint8)`.
  - Final array **must** be `[50,224,224,3]`, **uint8 / integers 0–255**.
- [ ] **Save:** `np.save("prediction.npy", arr)`. (Ilias renames upload to `.sec` — ignore. See dummy `dummy_colorizationnpy.sec` = shape `(50,224,224,3)` uint8 0–255.)

### Scoring (MSE, lower = better)
- MSE **< 45** → 3 points
- MSE **[45–55]** → 2 points
- MSE **[55–65]** → 1 point

Evaluation reference code:
```python
import numpy as np, os
from PIL import Image
label_root = "student_dataset/test_color_rgb/"   # only tutors have this folder
rgbimgs = sorted(os.listdir(label_root))
rgb_labels = np.stack([Image.open(os.path.join(label_root+img)) for img in rgbimgs], axis=0)
student_prediction = np.load("....npy")           # must be shape [50,224,224,3]
mse = np.square(np.subtract(student_prediction, rgb_labels)).mean()
```

---

## 3. Project B — Traffic Light Detection (TLD)

**Goal:** 2D object detector that, given an image, **detects and classifies** traffic lights.
Dataset: **ATLAS** (recorded by KIT autonomous vehicle CoCar NextGen), **YOLO format** already. (DTLD from Uni Ulm optionally usable but needs heavy preprocessing — ATLAS alone is sufficient.)
**25 classes** (index → name):

```
0 circle_green            13 arrow_right_green
1 circle_red              14 arrow_right_yellow
2 off                     15 arrow_straight_green
3 circle_red_yellow       16 arrow_straight_left_green
4 arrow_left_green        17 arrow_straight_red_yellow
5 circle_yellow           18 arrow_straight_left_red
6 arrow_right_red         19 arrow_straight_left_yellow
7 arrow_left_red          20 arrow_straight_left_red_yellow
8 arrow_straight_red      21 arrow_straight_right_red
9 arrow_left_red_yellow   22 arrow_straight_right_red_yellow
10 arrow_left_yellow      23 arrow_straight_right_yellow
11 arrow_straight_yellow  24 arrow_straight_right_green
12 arrow_right_red_yellow
```

### Tasks
- [ ] Get ATLAS dataset (YOLO format).
- [ ] Train a generic detector — **YOLO (Ultralytics)** or **RT-DETR** recommended (little training code; follow the Ultralytics tutorial). Use train-settings cfg for optimizer/LR/augmentation; defaults are a good start.
- [ ] Run inference on the **425 test images** (CoCar NextGen).
- [ ] Tune **confidence threshold + NMS** (see scoring note below).
- [ ] **Produce `predictions.csv`** with columns: `ImageName,xywh,Conf,Classification`
  - `ImageName`: test image filename.
  - `xywh`: `"[x, y, w, h]"` (quoted list). **x,y = center** of bbox; **w,h = total width/height**; all in **absolute pixels**.
  - `Conf`: model confidence (if model has none, fill constant e.g. `0.8` — confidence not used in scoring).
  - `Classification`: integer class index (see table).
  - **One row per bounding box.** Multiple boxes in an image → multiple rows. **No prediction for an image → no row** for it.
  - Do **not** reorder/alter test set.

Example (matches `dummy_tld_predictions.csv`):
```
ImageName,xywh,Conf,Classification
1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.93408203125,1
1708418258713137499_front_medium.jpg,"[1313.0, 289.75, 34.0, 98.5]",0.93359375,1
```

### Scoring (F1, higher = better) — `F1 = 2·TP / (2·TP + FP + FN)`
- TP = predicted box with **IoU > 0.5** vs a label box **and** matching class.
- FP = prediction with no matching ground truth.
- FN = visible traffic light in image not matched by any prediction.
- **F1 > 40** → 3 points · **[30–40]** → 2 points · **[20–30]** → 1 point

**Conf/NMS tip:** scored on F1. Low confidence threshold → recall ↑ / precision ↓; high → opposite. Traffic lights rarely overlap → use **aggressive NMS** to cheaply drop redundant boxes.

---

## 4. Submission documents (Ilias)

Upload **≥3 separate documents — NO bundled `.zip`.**

1. **`erklaerung.txt`** (Explanation) — short description of program + approach. Must include:
   - Which project you chose.
   - Which ML architecture you used.
   - Rough overview of how you trained.
2. **Predictions of the test set:**
   - Colorization → `.npy` (becomes `.sec` in Ilias — fine). Match dummy format.
   - TLD → `.csv`. Match dummy format.
   - Note: Ilias auto-converts unknown formats to `.sec` and back — ignore.
3. **Code** — all files used. Any language/framework; `.py` or notebooks. Code not explicitly graded, no required structure.

---

## 5. Self-check website

Reads your `.npy` (Colorization) or `.csv` (TLD) and shows the achieved score + possible bonus points:
**https://kit-ml2.streamlit.app/**

> **Action:** validate your output file here before the final Ilias upload.

---

## 6. Tutorials & compute

**PyTorch:** Beginner tutorial (NN + CNN chapters), Custom Datasets, optimizers & LR schedules, save/load model, layers/activations, datasets & dataloaders docs.
**Ultralytics YOLO:** train detectors in a few lines — YouTube tutorial https://www.youtube.com/watch?v=r0RspiLG260 ; train-settings https://docs.ultralytics.com/usage/cfg/#train-settings
**Google Colab (free GPU):** useful if no local GPU.
- Upload files: `from google.colab import files; uploaded = files.upload()` (must re-upload on each restart; the two `.zip`s are ~400MB).
- Better: put data in Google Drive: `from google.colab import drive; drive.mount('/content/drive')`. Best: unzip locally and upload the folders to Drive (avoids re-unzipping). `!unzip <path>` extracts in Colab.
**LLMs** (ChatGPT/Gemini/Copilot): can help write training pipelines / custom datasets.
**Reference implementations** (tutor-found via Google, correctness not guaranteed): colorization with GANs; colorization with CNN + U-Net style; CNN in U-Net style; U-Net/GAN.

---

## 7. Working checklist

- [ ] Form team (≤3), team lead adds members in Ilias by `u****` ID before deadline.
- [ ] Decide project (A Colorization / B TLD).
- [ ] Get dataset.
- [ ] Build data pipeline.
- [ ] Train model (GPU/Colab if needed).
- [ ] Run inference on test set in original order.
- [ ] Format output to match dummy file exactly.
- [ ] Validate score on https://kit-ml2.streamlit.app/
- [ ] Write `erklaerung.txt`.
- [ ] Upload ≥3 docs (explanation + predictions + code), no zip, before 31.07.2026 23:55.
