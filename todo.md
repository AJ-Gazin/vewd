# Vewd — TODO

## Make `output` a triggerable, selection-driven socket

**Goal:** turn the `output` socket into a triggerable state that emits a *user-selected subset* of the accumulated grid images, instead of just passing through the first wired input.

**Usage scenario:**
1. Accumulate ~10 images from different seeds on the same prompt (grid fills up across runs).
2. Review them as thumbnails in the Vewd node.
3. Select the subset worth upscaling.
4. Trigger `output` to pass exactly that subset downstream into an upscale workflow.

**Notes / open questions for the implementation session:**
- Needs a "send selection to output" action (button) + a way to re-run downstream with the chosen batch.
- Selected images are different seeds of the same prompt → likely same resolution → can batch into one IMAGE tensor for output. Confirm/handle mismatched sizes.
- Today's behavior: `output` auto-passes the first wired input every run (no user action). New behavior must be opt-in so it doesn't break the existing passthrough/`capture` modes.
- Selection state already exists in the grid (`state.selected`); reuse it. Backend already has `/vewd/set_batch` (`_batch_store`) which `process()` reads when no input is wired — this is the likely hook.
- Decide the trigger UX: does pressing the button queue a run automatically, or just arm the selection for the next manual run?

**Deferred to a separate session (per 2026-06-13 discussion).**
