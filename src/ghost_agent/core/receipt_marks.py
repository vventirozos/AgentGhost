"""§4QO — ONE home for the caveat literals a tool result carries.

The tool that emits a marker imports it from here, and any code that reads
it (the image notes, a future check) matches
it from here — two copies of one literal is how a check goes dark
(`reply_shape_check.FALLBACK_HEADS`). The turn RECORD that first motivated
this module was measured null and removed (journal §4QO); the literals stay.
Nothing here imports anything.
"""

#: image_generation: the model has NOT looked at the picture it produced
UNSEEN_IMAGE_NOTE = "You have NOT seen this image: do not claim any specific detail is in it."
#: image_generation: the SUCCESS head of a render (`_turn_generated_images` reads it too)
IMAGE_SUCCESS_HEAD = "SUCCESS: Image generated"
#: image_generation: the picture WAS looked at (the collage check's scene, §4QJ; the
#: render look, §4QP) — the blind second-render guard counts it as inspected
IMAGE_SEEN_HEAD = "WHAT THE PICTURE SHOWS (the vision model looked at it): "
#: §4QX: image_generation: the SYSTEM shows the look line in the delivered reply — the
#: sentence the tool writes right after that line (own paragraph); `agent._owned_look_note`
#: appends the line only when it finds this right after it
LOOK_OWNED_NOTE = "The SYSTEM shows the user that line itself, directly under the picture."
#: §4QX: the label the system writes before the look line in the delivered reply
LOOK_OWNED_LABEL = "What the picture shows: "
#: image_generation: the collage ladder's "do not render again" notes open with this — the blind
#: second-render guard refuses a re-roll the tool itself said is the user's call (§4QP r2 M2)
NO_RERENDER_NOTE = "Do NOT render again"
#: image_generation: the machine-readable form of that hold — ONE line of its own, before the
#: markdown line, written only by the ladder (`_plain` keeps a filename from spelling it; the
#: scene sentence follows the markdown line and is never searched) — §4QP r3 M1
NO_RERENDER_MARK = "[no re-render:"
#: introspect: the overview is not a health check
OVERVIEW_HEALTH_LINE = (
    "Health: NOT checked by this overview — nothing below is a system health "
    "check. Only system_utility(action='check_health') reports the live checks; "
    "without it, do not call the system healthy, green or fine.")
#: system_utility(check_health): the first line of a live health check
HEALTH_CHECK_HEAD = "--- SYSTEM HEALTH DIAGNOSTICS ---"
#: file_system write/append of a document: cited links this request never read
LINK_CHECK_HEAD = "LINK CHECK: "
#: the banner the turn loop prepends to a failed tool result it records
FAILURE_BANNER = "[FAILURE BANNER]"
