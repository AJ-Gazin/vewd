# CLAUDE.md — vewd

ComfyUI custom node: an image/media viewer (`Vewd`) that shows a thumbnail grid of
workflow outputs, with save/export/heart tooling. A sibling repo `../vewdtab` is the
same idea as a right-side panel (no canvas node, so it can only auto-capture).

## Fork / local-patch model
This repo is upstream-maintained by `spiritform`; we run a fork (`AJ-Gazin/vewd`).
- Remotes: `origin` = the fork (push here), `upstream` = spiritform (source of truth).
- Update flow: `git fetch upstream && git rebase upstream/main`, resolve conflicts at
  the `Local patch` sentinels, `git push origin main --force-with-lease`, restart ComfyUI.
- **Wrap every local change in `--- Local patch: <name> ---` … `--- end local patch ---`
  (or `// Local patch:` in JS) so it's greppable during rebases.** `grep -rn "Local patch"`.

## Architecture — the one thing to internalize
There are **two independent mechanisms**, and confusing them causes most bugs:

1. **Python node (`nodes.py` `Vewd.process`)** runs at graph-execution time. It only
   produces the `IMAGE` passthrough output and an optional `ui` dict. It does **not**
   populate the grid (except by emitting a private `ui` key the frontend reads).
2. **Frontend (`web/vewd.js`)** owns the grid. It listens to the global ComfyUI
   `executed` event and adds *any* node's image outputs to the grid. **Saving/exporting
   is also frontend-driven**: the `save`/`export` buttons POST to `/vewd/save` and
   `/vewd/export` using the DOM field values (`folderInput.value`, `prefixInput.value`) —
   the Python `folder`/`filename_prefix` params are only defaults that seed those fields.

So: to change *what shows in the grid* → frontend listener. To change *how files are
named/written* → the HTTP route handlers in `nodes.py`. To change *the passthrough
tensor* → `Vewd.process`.

There is a single global grid widget (`globalVewdWidget`); multiple Vewd nodes share it.

## Gotchas (learned the hard way)
- **`OUTPUT_NODE = True` is required** for the node to be a terminal. Without it, a branch
  wired *only* into Vewd's input gets pruned by ComfyUI's demand-driven executor and never
  runs. Vewd behaves like Preview/Save Image.
- **`ui` values must be lists.** `{"ui": {"key": "abc"}}` gets coerced to `list("abc")` =
  `['a','b','c']`. Always emit `{"ui": {"key": [value]}}` and unwrap `[0]` on the frontend.
- **Custom `ui` keys avoid native previews.** Returning `ui.images` makes ComfyUI draw its
  own under-node image preview. To feed the grid *without* that, emit under a private key
  (we use `vewd_images`) and handle it in the frontend listener.
- **Append new inputs at the END of `INPUT_TYPES`.** Widget values in saved workflows are
  positional; inserting an input mid-list shifts them and corrupts later widgets
  (e.g. `max_frames` gets `capture`'s value). `forceInput` STRING = a socket, not a widget.
- **`nodeCreated` runs before a loaded workflow is configured.** `node.id` and restored
  widget values aren't set yet — do **not** cache them there. Keep a live `node` reference
  (`widget._node`) and read `node.id` / widget values at event time.
- The widgets `folder`, `filename_prefix`, `selected_media` are spliced out of
  `node.widgets` in `nodeCreated` and rendered as DOM inputs instead.

## Features that exercise the above (see git log + `Local patch` blocks for detail)
- `capture` combo: `auto` (scrape whole workflow) vs `input` (only this node's wired
  tensors). Input mode emits `vewd_images`; the listener guard drops other nodes.
- Multi-input accumulate sockets `input`/`input_2..4`; passthrough output = first connected.
- Wire-driven `prefix` socket → `vewd_prefix` ui round-trip → fills the prefix field;
  the field is disabled while the socket is wired.
- Save naming: `/vewd/save` writes `prefix_NNN.ext` (seed kept in PNG metadata, not the
  name). `/vewd/export` (heart/tag → `/selects`) still uses `prefix_seed_NNN` — intentional.

## Verifying changes
No automated tests for the node UI. After editing, restart ComfyUI and run a workflow;
watch the server console for `[Vewd]` logs and the browser console for `[Vewd]` warnings.
`todo.md` tracks planned work (e.g. selection-driven triggerable output).
