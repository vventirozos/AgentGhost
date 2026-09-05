"""Ask-about-this-file on upload (2026-09-05, operator):

    "when i click upload and i take a picture, the picture gets uploaded
     but in order to ask something about the picture i have to type the
     filename (image.jpg) — can we have a window that will include a
     request? like 'what do you see', or 'what kind of car is that'?"

Choosing a file now opens a sheet (preview for images, the stored name, a
question box). "Upload & ask" uploads, then sends question + stored
filename as ONE chat message through the ordinary send path; "Upload
only" keeps the old behaviour. iOS names every camera capture image.jpg,
so generic capture names get a timestamp (a second photo no longer
overwrites the first).

Pure helpers are EXECUTED under node; the sheet's geometry is RENDERED in
WebKit at phone width with the real markup + CSS; the wiring is text.
"""

import re
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments
from tests.test_interface_header_simplify import playwright_browser, _open, _boxes  # noqa: F401

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


def _raw(name):
    return (_STATIC / name).read_text(encoding="utf-8")


def _js(name):
    return strip_js_comments(_raw(name))


class TestHelpers:

    @pytest.fixture(scope="class")
    def app_js(self):
        return _js("app.js")

    def test_compose_puts_the_question_first_and_names_the_stored_file(self, app_js):
        fn = extract_js_function(app_js, "composeUploadRequest")
        out = eval_js(fn, "composeUploadRequest('  What kind of car is that?  ', 'photo-20260905-094012.jpg')")
        assert out == "What kind of car is that?\n\n(File just uploaded to the sandbox: photo-20260905-094012.jpg)"
        assert out.index("What kind of car") < out.index("photo-20260905")

    def test_an_empty_question_composes_nothing(self, app_js):
        fn = extract_js_function(app_js, "composeUploadRequest")
        for q in ("''", "'   '", "null", "undefined"):
            assert eval_js(fn, f"composeUploadRequest({q}, 'a.jpg')") == "", q

    def test_generic_camera_names_get_a_timestamp_others_keep_theirs(self, app_js):
        src = ("const _GENERIC_CAPTURE_NAMES = " + re.search(r"const _GENERIC_CAPTURE_NAMES = (/.*?/i);", app_js).group(1) + ";\n"
               + extract_js_function(app_js, "uploadNameFor"))
        clock = "new Date(2026, 8, 5, 9, 40, 12)"
        cases = {
            "image.jpg": "photo-20260905-094012.jpg",
            "IMAGE.JPEG": "photo-20260905-094012.jpeg",
            "image.png": "photo-20260905-094012.png",
            # A Photos-library pick (IMG_0042) is already unique: kept.
            "IMG_0042.HEIC": "IMG_0042.HEIC",
            "IMG42.heic": "IMG42.heic",
            "capture7.JPG": "photo-20260905-094012.jpg",
            "photo.webp": "photo-20260905-094012.webp",
            "holiday-car.jpg": "holiday-car.jpg",
            "report.pdf": "report.pdf",
            "image.txt": "image.txt",
            "my image.jpg": "my image.jpg",
        }
        for name, want in cases.items():
            got = eval_js(src, f"uploadNameFor({{name: {name!r}}}, {clock})")
            assert got == want, f"{name} -> {got!r}, want {want!r}"
        assert eval_js(src, "uploadNameFor({}, " + clock + ")") == "upload"

    def test_images_open_with_the_default_question_other_files_empty(self, app_js):
        src = ("const UPLOAD_ASK_IMAGE_DEFAULT = 'Describe this image:';\n"
               + extract_js_function(app_js, "uploadAskDefaultFor"))
        cases = {"image/jpeg": "Describe this image:", "image/png": "Describe this image:",
                 "image/heic": "Describe this image:", "application/pdf": "", "text/plain": "", "": ""}
        for mime, want in cases.items():
            assert eval_js(src, f"uploadAskDefaultFor({{type: {mime!r}}})") == want, mime
        assert eval_js(src, "uploadAskDefaultFor({})") == ""
        assert eval_js(src, "uploadAskDefaultFor(null)") == ""

    def test_editing_a_pristine_default_replaces_it_or_clears_it(self, app_js):
        fn = extract_js_function(app_js, "resolvePristineInput")
        D = "Describe this image:"
        cases = [
            (D, D),                                   # untouched
            (D + " what colour is the car", "what colour is the car"),   # dictated/pasted after it
            (D + "W", "W"),                           # a character typed after it
            ("Describe this image", ""),              # backspace ate one char → clear all
            ("Describe", ""),                         # several deleted → clear all
            ("", ""),                                 # cleared
            ("W", "W"),                               # replaced outright
            ("what is this?", "what is this?"),
        ]
        for value, want in cases:
            got = eval_js(fn, f"resolvePristineInput({D!r}, {value!r})")
            assert got == want, f"{value!r} -> {got!r}, want {want!r}"
        assert eval_js(fn, "resolvePristineInput('', 'hello')") == "hello"

    def test_the_first_key_on_a_pristine_default_replaces_or_clears(self, app_js):
        js = app_js
        kd = js[js.index("uploadAskInput?.addEventListener('keydown'"):]
        kd = kd[:kd.index("});") + 3]
        assert "if (!uploadAskPristine || e.metaKey || e.ctrlKey || e.altKey) return;" in kd
        assert "if (e.key === 'Backspace' || e.key === 'Delete') {" in kd
        assert "e.preventDefault();" in kd and "uploadAskInput.value = '';" in kd
        assert "} else if (e.key.length === 1) {" in kd, "a typed character must replace the default"
        assert kd.count("_setUploadAskPristine(false);") == 2
        # the sheet opens with the default and the pristine flag; input events resolve edits
        assert "uploadAskInput.value = uploadAskDefaultFor(file);" in js
        assert "_setUploadAskPristine(uploadAskInput.value !== '');" in js
        assert "const resolved = resolvePristineInput(UPLOAD_ASK_IMAGE_DEFAULT, uploadAskInput.value);" in js
        assert ".modal-textarea.pristine" in _raw("style.css")

    def test_bytes_formatter(self, app_js):
        fn = extract_js_function(app_js, "_fmtBytes")
        assert eval_js(fn, "[_fmtBytes(0), _fmtBytes(999), _fmtBytes(20480), _fmtBytes(3.5 * 1024 * 1024)]") == \
            ["0 B", "999 B", "20 KB", "3.5 MB"]


class TestWiring:

    def test_choosing_a_file_opens_the_sheet_instead_of_uploading(self):
        js = _js("app.js")
        change = js[js.index("fileUploadInput.addEventListener('change'"):js.index("const _GENERIC_CAPTURE_NAMES")]
        assert "openUploadAsk(file);" in change
        assert "fetch('/api/upload'" not in change, "the change handler still uploads blind"
        assert "fileUploadInput.value = '';" in change

    def test_one_button_uploads_and_an_empty_box_means_upload_only(self):
        js = _js("app.js")
        fn = extract_js_function(js, "confirmUploadAsk")
        assert "const question = uploadAskInput ? uploadAskInput.value : '';" in fn
        assert "const stored = await uploadToSandbox(file, storedName);" in fn
        assert "const message = composeUploadRequest(question, stored);" in fn
        assert "if (!message) return;" in fn, "an empty question must upload and stop — no message"
        assert "chatInput.value = message;" in fn and "await sendMessage();" in fn
        assert fn.index("closeUploadAsk();") < fn.index("uploadToSandbox("), "the sheet must close before the upload runs"
        assert "uploadAskSend?.addEventListener('click', () => confirmUploadAsk());" in js
        assert "upload-ask-only" not in js and "confirmUploadAsk(true)" not in js
        # The input listener resolves a pristine default first, then syncs the label.
        inp = js[js.index("uploadAskInput?.addEventListener('input'"):]
        inp = inp[:inp.index("});") + 3]
        assert "_syncUploadAskLabel();" in inp and "resolvePristineInput(" in inp

    def test_the_button_label_follows_the_box(self):
        fn = extract_js_function(_js("app.js"), "uploadAskLabel")
        assert eval_js(fn, "[uploadAskLabel(''), uploadAskLabel('   '), uploadAskLabel(null), uploadAskLabel('what is this?')]") == \
            ["Upload", "Upload", "Upload", "Upload & ask"]

    def test_the_upload_uses_the_stored_name_and_reports_the_servers_name(self):
        fn = extract_js_function(_js("app.js"), "uploadToSandbox")
        assert "formData.append('file', file, storedName);" in fn
        assert "typeof result.filename === 'string'" in fn
        assert "return stored;" in fn and "return null;" in fn
        assert "isProcessingRequest = false;" in fn[fn.index("finally"):]

    def test_close_paths_revoke_the_preview(self):
        js = _js("app.js")
        fn = extract_js_function(js, "closeUploadAsk")
        assert "URL.revokeObjectURL(pendingUpload.previewUrl)" in fn
        assert "pendingUpload = null;" in fn
        for path in ("upload-ask-close')?.addEventListener('click', closeUploadAsk)",
                     "if (e.target === uploadAskModal) closeUploadAsk();",
                     "e.key === 'Escape' && !uploadAskModal.classList.contains('hidden')) closeUploadAsk();"):
            assert path in js, path
        assert "if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); confirmUploadAsk(); }" in js

    def test_markup(self):
        html = _raw("index.html")
        for el_id in ("upload-ask-modal", "upload-ask-box", "upload-ask-preview", "upload-ask-file",
                      "upload-ask-input", "upload-ask-send", "upload-ask-close"):
            assert f'id="{el_id}"' in html, el_id
        sheet = html[html.index('id="upload-ask-modal"'):html.index('id="memory-modal"')]
        assert 'class="modal-overlay hidden"' in sheet, "must start hidden"
        assert 'for="upload-ask-input"' in sheet, "the question box needs a label"
        assert 'enterkeyhint="send"' in sheet
        assert 'id="upload-ask-only"' not in sheet and "Upload only" not in sheet, "the second button is back"
        assert sheet.count('class="modal-btn') == 1, "exactly one action button"
        # Operator (same day): the box comes FIRST, the picture is a thumbnail
        # beside the filename, and there are no example prompts in it.
        assert sheet.index('id="upload-ask-input"') < sheet.index('id="upload-ask-preview"')
        ta = re.search(r"<textarea id=\"upload-ask-input\"[^>]*>", sheet).group(0)
        assert "placeholder=" not in ta, "the example prompts are back in the box"
        # The image default is set by app.js at open time (it depends on the
        # file type), never baked into the markup.
        assert "Describe this image" not in sheet
        assert 'id="upload-ask-meta"' in sheet


class TestRendered:

    def test_the_sheet_fits_a_phone_with_the_keyboard_row_reachable(self, playwright_browser):
        ctx, page = _open(playwright_browser, viewport={"width": 390, "height": 844},
                          has_touch=True, is_mobile=True)
        try:
            page.evaluate("""() => {
                const m = document.getElementById('upload-ask-modal'); m.classList.remove('hidden');
                document.getElementById('upload-ask-preview-wrap').hidden = false;
                document.getElementById('upload-ask-preview').src =
                    'data:image/svg+xml;utf8,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1600"><rect width="1200" height="1600" fill="#345"/></svg>');
                document.getElementById('upload-ask-file').textContent = 'image.jpg → photo-20260905-094012.jpg · 2.1 MB';
            }""")
            page.wait_for_timeout(150)
            boxes = _boxes(page, ["upload-ask-box", "upload-ask-preview", "upload-ask-input", "upload-ask-send", "upload-ask-file"])
            box, preview, inp, send, fileline = boxes
            assert 0 <= box["left"] and box["right"] <= 390 and box["bottom"] <= 844, box
            for b in (preview, inp, send):
                assert b["left"] >= box["left"] and b["right"] <= box["right"] + 0.5, b
            assert send["h"] >= 44
            assert inp["h"] >= 40
            # The QUESTION is the subject: box above the picture, and the whole
            # sheet — buttons included — sits above where the iPhone keyboard
            # lands (~336px of an 844px screen), so nothing hides under it.
            assert inp["top"] < preview["top"], "the picture is above the question box again"
            assert preview["w"] <= 80 and preview["h"] <= 80, f"the preview is not a thumbnail: {preview}"
            assert abs(preview["top"] + preview["h"] / 2 - (fileline["top"] + fileline["h"] / 2)) < 12, "thumbnail not beside the filename"
            assert send["bottom"] <= 844 - 336, f"the buttons would sit under the keyboard: {send}"
            assert box["top"] <= 40, "the sheet must be top-anchored on a phone"
            assert send["bottom"] <= box["bottom"] + 0.5
        finally:
            ctx.close()

    def test_hidden_by_default_and_desktop_width_is_bounded(self, playwright_browser):
        ctx, page = _open(playwright_browser, viewport={"width": 1280, "height": 800})
        try:
            assert page.evaluate("getComputedStyle(document.getElementById('upload-ask-modal')).display") == "none"
            page.evaluate("document.getElementById('upload-ask-modal').classList.remove('hidden')")
            box = _boxes(page, ["upload-ask-box"])[0]
            # min(520px, 100%) content box + 1px borders = 522, same as the memory modal.
            assert box["w"] <= 522 + 0.5 and abs((box["left"] + box["right"]) / 2 - 640) < 2, box
        finally:
            ctx.close()
