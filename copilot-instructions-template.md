# [Project Name] — AI Agent Instructions (Template)

> **Purpose:** A reusable template for `.github/copilot-instructions.md`.
> Copy this file into a new project's `.github/` folder and fill in the
> project-specific details marked with `[PLACEHOLDER]`.
>
> **Last updated:** [DATE]
> Read this file at the start of every new chat session.
> Update this file when new conventions, preferences, or project structure changes are established in chat.

---

## 1. Project Overview

[One-paragraph description of the project's goal, domain, and primary outputs.]

### Workspace layout

```
project-root/
├── .github/copilot-instructions.md   ← THIS FILE
├── src/                               ← shared Python modules
│   ├── __init__.py
│   └── [shared_module].py            ← shared labels, constants, helpers
├── notebooks/
│   ├── [main-notebook].ipynb          ← primary analysis / figures
│   ├── [secondary-notebook].ipynb     ← data processing / derived columns
│   └── archive/                       ← old or dated notebooks
├── figures/
│   └── [batch_name]/                  ← one folder per batch / experiment
│       ├── fig1_*.png                 ← publication figures
│       ├── figure_registry.txt        ← full registry (auto-generated)
│       ├── figure_registry_concise.txt← concise registry (auto-generated)
│       └── scratch/                   ← exploratory figures (never registered)
├── latex/                             ← auto-generated LaTeX compilations
│   ├── build_figure_doc.py
│   ├── watch_registry.sh
│   └── [batch_name]_figures.tex       ← generated (do not hand-edit)
└── [other_code]/                      ← legacy or domain-specific modules
```

### Key variables

| Python name | Meaning |
|---|---|
| `[main_dataframe]` | Primary DataFrame (one row per [unit of analysis]) |
| `rename` | Column → LaTeX label dict (defined in shared module) |
| [Add project-specific variables here] |

---

## 2. Notebook Conventions

### Imports & sys.path

[Describe how notebooks find shared code — e.g., sys.path manipulation, package installs, etc.]

```python
# Example pattern:
import sys as _sys
_sys.path.insert(0, "/absolute/path/to/src")
from [shared_module] import (rename, renameit, FS_LABEL, FS_TITLE, FS_TICK, FS_LEG, VAR_CMAPS, ...)
```

### Python environment

- Python: `[path or version]`
- Key packages: [numpy, pandas, matplotlib, seaborn, ...]
- Virtual env: [describe or "none required"]

### Cell structure preferences

- Keep one logical task per cell.
- Markdown cells before major sections.
- Use `df.copy()` before in-place mutations to avoid corrupting shared state.
- Prefer `_private` names (leading underscore) for cell-local temporaries.

### Feature toggles

[Document any boolean flags that change analysis behavior.]

```python
# Example:
USE_ALTERNATE = False   # When True, overwrite X with Y
```

---

## 3. Shared Code — `src/[shared_module].py`

All display labels, colour maps, and formatting helpers shared between notebooks live in `src/[shared_module].py`. **Do not duplicate these in notebook cells.**

### Exports

| Name | Type | Purpose |
|---|---|---|
| `rename` | dict | Column → LaTeX label mappings |
| `renameit(name, mapping)` | function | Safe lookup: returns `name` if key missing |
| `FS_LABEL, FS_TITLE, FS_TICK, FS_LEG` | int | Global matplotlib font sizes |
| `VAR_CMAPS` | dict | Per-variable canonical matplotlib colormaps |
| `[CATEGORY]_COLORS` | dict | Canonical hex colours per category |
| `[CATEGORY]_LABELS` | dict | Display labels per category |

### Adding new labels

When a new column appears in the main DataFrame, add its display label to the shared module's `rename` dict — not in a notebook cell.

### Colourmap assignments

Every [parameter/variable] has a fixed colourmap. Always use `VAR_CMAPS.get(var, 'viridis')` when colouring by a variable.

[List the current assignments in a table:]

| Variable | Colourmap |
|---|---|
| [var1] | [cmap1] |
| [var2] | [cmap2] |
| ... | ... |

---

## 4. Figures & Figure Registry

### Two-tier figure system

1. **Publication figures** → saved to `figures/[batch]/`, named `fig<N>_<description>.png`
2. **Scratch/exploratory figures** → saved to `figures/[batch]/scratch/`, **never registered**

### Saving a publication figure

Every publication figure save must call `update_figure_registry()`:

```python
_fig_dir, _, _ = _fig_dirs()
_name = 'fig1_descriptive_name.png'
fig.savefig(_os.path.join(_fig_dir, _name), dpi=300, bbox_inches='tight')
update_figure_registry(
    'fig1', _name,
    description='Full multi-line description...',
    concise='Two-sentence summary. Key takeaway.')
```

### Saving a scratch figure

```python
_, _scratch, _ = _fig_dirs()
fig.savefig(_os.path.join(_scratch, 'name.png'), dpi=200, bbox_inches='tight')
# NO registry call
```

### Registry behaviour

- Accepts: `fig_id`, `filename`, `description` (full), `concise` (2 sentences: what + interpretation)
- Automatically re-sorts entries: main figures (fig1 → figN) first, SI figures at end
- Writes **two files** on every call:
  - `figure_registry.txt` — full detailed registry with metadata
  - `figure_registry_concise.txt` — short human-readable version
- Both files include: creation date, source notebook, figure save directory
- **Descriptions should be computed dynamically** from plot metrics (R², RMSE, etc.) so they stay current when data changes

### Registry entry format (full)

```
### fig1 ###
File     : fig1_descriptive_name.png
Updated  : YYYY-MM-DD HH:MM
Notebook : notebooks/[notebook].ipynb
Saved in : figures/[batch]
────────────────────────────────────────
<description with interpretation>
### end fig1 ###
```

### Registry entry format (concise)

```
Fig 1 — fig1_descriptive_name.png
  <Two sentences: description + key finding.>
```

### Figure numbering

- Main figures: `fig1`, `fig2`, `fig3`, … (sequential, in paper order)
- SI figures: `SI1`, `SI2`, … (sorted after main figures)
- Numbering should be stable; don't renumber existing figures without explicit request

### Figure style rules

- Use font-size constants from the shared module (`FS_LABEL`, `FS_TITLE`, etc.)
- Use `VAR_CMAPS` when colouring by a variable
- Use category colour/label dicts for categorical variables
- Metric annotations (RMSE, R², etc.): `ax.text()` in a corner, not in the title
- Legend style: prefer `Line2D` circle markers over `Patch` rectangles for scatter legends
- Truncate colormaps (e.g., 0.2–0.95) to avoid washed-out extremes near white
- Add small y-jitter when observed values cluster on discrete levels
- Use `renameit()` for axis labels whenever possible

---

## 5. LaTeX Figure Compilations

Each batch folder in `figures/` gets its own auto-generated LaTeX document.

### How it works

1. `latex/build_figure_doc.py` reads `figures/[batch]/figure_registry_concise.txt`
2. Generates `latex/[batch]_figures.tex` with one `\figure` per registry entry
3. Compiles to PDF with `pdflatex` (if installed); exits cleanly if not

### Usage

```bash
# One-shot build (all batches, or specify one)
python3 latex/build_figure_doc.py
python3 latex/build_figure_doc.py [batch_name]

# Auto-rebuild on registry changes (requires: brew install fswatch)
./latex/watch_registry.sh
```

### Conventions

- **Do not hand-edit** generated `.tex` files — they are overwritten on every build
- Figure labels match registry IDs: `\label{fig:fig1}`, `\label{fig:SI1}`, etc.
- One document per batch folder
- Concise registry caption → `\caption{…}`; empty captions fall back to filename
- New batch folders are auto-discovered when the build script runs with no args

---

## 6. Notebook File Management

- **Active notebooks** live in `notebooks/`.
- **Archived notebooks** go to `notebooks/archive/` with a date suffix if not already dated.
- Do not create new notebooks without being asked — modify existing ones.

---

## 7. Editing Preferences

- When editing `.ipynb` files, use `replace_string_in_file` with exact text matching (include 3–5 lines of context).
- For complex multi-line notebook cell edits that fail with text matching, fall back to a Python script using `json.load` / `json.dump`.
- Always use `df.copy()` before any operation that modifies a subset in-place.
- Prefer `_private` names (leading underscore) for cell-local temporaries to avoid polluting the kernel namespace.

---

## 8. Updating This File

**This file should be updated whenever:**
- A new shared convention is established (colour, font, label, file layout)
- A new figure is added to the registry numbering scheme
- LaTeX or manuscript conventions are established
- New shared code is added to `src/`
- File organization rules change

Add new sections or update existing ones. Keep the format consistent.
