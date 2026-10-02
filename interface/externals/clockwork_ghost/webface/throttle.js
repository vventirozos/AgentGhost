// Frame-rate cap for the handheld's face (2026-10-01).
//
// Measured on the device with nothing happening: the client 44.5% of a core,
// the web renderer 29.9% — the face redrawing at full rate behind glass that
// re-blends on every frame, for nobody. The canonical matrix_graph.js has a
// pause (setAnimationPaused) but no slow gear, and it must stay a byte-for-
// byte copy of the browser's file, so the gear goes UNDERNEATH it: this wraps
// requestAnimationFrame before the face module loads.
//
// A capped frame waits on a timer and only then asks for a real animation
// frame, so the renderer sleeps between frames instead of waking 60 times a
// second to decline to draw. cancelAnimationFrame is wrapped too: the id the
// face holds is OUR id, and its own pause (which cancels that id) would
// otherwise cancel nothing and the loop would keep running.
//
// Two things a cap does that are not obvious (measured in Chromium against
// the real face.html):
//   * It is a CEILING, and the rate lands a little under it — the timer is
//     followed by a wait for the next display refresh: cap 10 draws ~9.2
//     frames a second, cap 5 ~4.8.
//   * The face runs in SLOW MOTION, not just choppily. matrix_graph.js steps
//     its animation by elapsed time but clamps one step at three 60 Hz
//     frames, so below 20 fps time itself slows: about half speed at cap 10,
//     a quarter at cap 5. That suits what the cap is for — a face nobody is
//     looking at — and is why activity lifts the cap rather than keeping it.
//
// A classic script, not a module: it has to run before the module graph.
function installThrottle(win) {
    const raf = win.requestAnimationFrame.bind(win);
    const caf = win.cancelAnimationFrame.bind(win);
    const setT = win.setTimeout.bind(win);
    const clearT = win.clearTimeout.bind(win);
    const now = () => win.performance.now();
    let cap = 0;            // frames per second; 0 = uncapped
    let last = -1e9;        // when the last frame ran
    let nextId = 1;
    const live = new Map(); // our id -> { raf } | { timer }

    win.requestAnimationFrame = (cb) => {
        const id = nextId++;
        const run = (ts) => {
            if (!live.delete(id)) return;      // cancelled while waiting
            last = now();
            cb(ts);
        };
        const arm = () => {
            if (live.has(id)) live.set(id, { raf: raf(run) });
        };
        const schedule = () => {
            const wait = cap > 0 ? 1000 / cap - (now() - last) : 0;
            if (wait > 4) live.set(id, { timer: setT(arm, wait), schedule });
            else arm();
        };
        live.set(id, {});
        schedule();
        return id;
    };

    win.cancelAnimationFrame = (id) => {
        const h = live.get(id);
        if (!h) return;
        live.delete(id);
        if (h.raf !== undefined) caf(h.raf);
        else clearT(h.timer);
    };

    const api = {
        // A frame already waiting on the OLD cap's timer is rescheduled, so
        // lifting the cap takes effect now and not up to a whole slow frame
        // later (200 ms at a cap of 5).
        setFps(n) {
            cap = Math.max(0, Number(n) || 0);
            for (const h of Array.from(live.values())) {
                if (h.timer !== undefined) { clearT(h.timer); h.schedule(); }
            }
            return cap;
        },
        getFps() { return cap; },
        pending() { return live.size; },
    };
    win.__faceThrottle = api;
    return api;
}
if (typeof window !== 'undefined') installThrottle(window);
