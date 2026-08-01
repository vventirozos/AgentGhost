"""Visualizer window (#render-window) contract pins — 2026-08-01 review.

The feature's recurring defect class is "window opens, shows nothing".
Root cause of the headline bug: PDFs rendered into a NESTED iframe inside
a sandbox'd iframe — the plugins-sandbox flag is unclearable, inherited,
and blocks the PDF viewer in every engine. These pins are source-level
(the feature is DOM-side; the established style for client invariants).
"""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
APP = (STATIC / "app.js").read_text()
HTML = (STATIC / "index.html").read_text()
CSS = (STATIC / "style.css").read_text()
SERVER = (Path(__file__).resolve().parents[1] / "interface" / "server.py").read_text()


class TestFrameAttributes:
    def test_main_frame_is_not_sandboxed(self):
        # sandbox on the same-origin frame is a security no-op that blocks
        # the PDF viewer. It must never come back.
        m = re.search(r'<iframe id="render-iframe"([^>]*)>', HTML)
        assert m, "render-iframe missing"
        assert "sandbox" not in m.group(1)

    def test_untrusted_frame_keeps_strict_sandbox(self):
        # ALL containment for agent-authored code lives here: allow-scripts
        # ONLY. allow-same-origin would hand agent code the API key.
        m = re.search(r'<iframe id="render-iframe-sandboxed"([^>]*)>', HTML)
        assert m, "render-iframe-sandboxed missing"
        assert 'sandbox="allow-scripts"' in m.group(1)
        assert "allow-same-origin" not in m.group(1)

    def test_agent_code_only_enters_the_sandboxed_frame(self):
        # renderHTMLContent must target the sandboxed frame via srcdoc.
        body = APP.split("function renderHTMLContent", 1)[1].split("\nfunction ", 1)[0]
        assert "renderIframeSandboxed.srcdoc" in body
        assert "renderIframe.srcdoc" not in body.replace("renderIframeSandboxed.srcdoc", "")


class TestPdfPath:
    def test_no_nested_iframe_pdf_write(self):
        assert "_writePdfIntoIframe" not in APP  # the sandboxed-nesting design
        # Call syntax only — the history lives on in comments deliberately.
        assert "contentDocument.write(" not in APP  # racy + can't follow a PDF src

    def test_desktop_gets_direct_blob_src_and_ios_gets_panel(self):
        assert "if (isIOS)" in APP
        assert "_showPdfPanel(_pdfBlobUrl" in APP
        assert "renderIframe.src = _pdfBlobUrl" in APP

    def test_pdf_bypasses_the_image_lru(self):
        # Multi-MB PDFs in a 100-entry cache = phone tab-kill. Dedicated
        # single URL, revoked on replace and on close.
        pdf_click = APP.split("_handleChatPdfLink", 1)[1].split("\nfunction ", 1)[0]
        assert "_toAuthedBlobUrl" not in pdf_click
        assert "_pdfBlobUrl = URL.createObjectURL" in APP
        close_body = APP.split("function closeRenderWindow", 1)[1].split("\n}", 1)[0]
        assert "_pdfBlobUrl" in close_body

    def test_pdf_click_no_longer_auto_downloads(self):
        pdf_click = APP.split("_handleChatPdfLink", 1)[1].split("\nfunction ", 1)[0]
        assert "a.download" not in pdf_click

    def test_download_button_has_a_pdf_branch(self):
        assert "currentRenderState.type === 'pdf'" in APP


class TestResetDiscipline:
    def test_reset_helper_exists_and_tears_down_everything(self):
        body = APP.split("function resetRenderSurfaces", 1)[1].split("\nfunction ", 1)[0]
        for needle in ("removeAttribute('srcdoc')", "replaceChildren()",
                       "currentChart.destroy()", "currentZoom = 1.0"):
            assert needle in body, f"reset helper lost: {needle}"

    def test_every_open_path_resets_first(self):
        # Each open site must call resetRenderSurfaces before showing its
        # surface — otherwise stale content/zoom from the previous artifact
        # leaks through (the 'opens on nothing' class).
        assert APP.count("resetRenderSurfaces()") >= 5

    def test_srcdoc_interpolations_are_escaped(self):
        # Every ${...} inside a srcdoc-building template must go through
        # _escAttr (attribute-breakout guard for the same-origin frame).
        for m in re.finditer(r"\$\{(?!_escAttr\()([^}]*)\}", APP):
            src_ctx = APP[max(0, m.start() - 400):m.start()]
            if "_setMainFrameHTML" in src_ctx or "srcdoc" in src_ctx.lower():
                raise AssertionError(
                    f"unescaped srcdoc interpolation: ${{{m.group(1)}}}")

    def test_clear_paths_close_the_window_before_revoking(self):
        # Revoking a blob the visualizer is displaying blanks the preview.
        clear_idx = APP.find("clearConversation")
        revoke_idx = APP.find("URL.revokeObjectURL", clear_idx)
        close_idx = APP.find("closeRenderWindow()", clear_idx)
        assert close_idx != -1 and close_idx < revoke_idx

    def test_lru_eviction_spares_the_displayed_blob(self):
        evict = APP.split("function _evictAuthedBlobCache", 1)[1].split("\nfunction ", 1)[0]
        assert "currentRenderState.src === oldestUrl" in evict


class TestCssAndErrors:
    def test_opaque_containers_hidden_by_default(self):
        # These paint OVER the iframes when visible; a single missed hide
        # reproduces 'window opens, shows nothing'.
        for sel in ("#mermaid-container", "#chart-container"):
            # The DEDICATED block = the LAST line-anchored occurrence (an
            # earlier combined `a,\nb {` rule also matches the pattern).
            block = CSS.rsplit("\n" + sel + " {", 1)[1].split("}", 1)[0]
            assert "display: none" in block, f"{sel} lacks display:none default"

    def test_no_important_on_mobile_render_window_override(self):
        # !important here beats the drag/resize inline styles — both were
        # silently dead on landscape phones and the uConsole.
        uconsole = CSS.split("max-width: 1280px", 1)[1].split("@media", 1)[0]
        rw = uconsole.split("#render-window", 1)[1].split("}", 1)[0]
        assert "!important" not in rw

    def test_failures_render_a_visible_panel(self):
        assert "_showRenderError" in APP
        csv_body = APP.split("function renderCSV", 1)[1].split("\nfunction ", 1)[0]
        assert "_showRenderError" in csv_body          # <2 columns / no vendor
        assert "window.Papa" in csv_body and "window.Chart" in csv_body


class TestDownloadProxyHardening:
    def test_nosniff_and_html_attachment(self):
        dl = SERVER.split('def download_proxy', 1)[1].split("\n@app.", 1)[0]
        assert "x-content-type-options" in dl
        assert "nosniff" in dl
        assert '"attachment"' in dl
