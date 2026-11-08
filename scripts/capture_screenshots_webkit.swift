import AppKit
import WebKit

final class ScreenshotDriver: NSObject, WKNavigationDelegate {
    private let output: URL
    private let window: NSWindow
    private let webView: WKWebView
    private var readinessAttempts = 0

    init(output: URL) {
        self.output = output
        let frame = NSRect(x: 0, y: 0, width: 1440, height: 1800)
        self.window = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        self.webView = WKWebView(frame: NSRect(origin: .zero, size: frame.size))
        super.init()
        webView.navigationDelegate = self
        window.contentView = webView
        window.setFrame(frame, display: true)
    }

    func start(url: URL) {
        print("Loading local Streamlit page: \(url.absoluteString)")
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        webView.load(URLRequest(url: url))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        print("WebKit navigation finished")
        waitUntilRendered()
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        fail("Navigation failed: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        fail("Provisional navigation failed: \(error.localizedDescription)")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        fail("WebKit content process terminated")
    }

    private func waitUntilRendered() {
        readinessAttempts += 1
        webView.evaluateJavaScript(
            "({text: document.body?.innerText || '', ready: !!document.querySelector('[role=tab]')})"
        ) { [weak self] result, error in
            guard let self else { return }
            if let error {
                self.fail("Page evaluation failed: \(error.localizedDescription)")
                return
            }
            let state = result as? [String: Any] ?? [:]
            let text = state["text"] as? String ?? ""
            let hasInference = text.contains("TinyCD device:") || text.contains("TinyCD inference failed")
            if (state["ready"] as? Bool == true) && text.contains("Qualitative demo only") && hasInference {
                DispatchQueue.main.asyncAfter(deadline: .now() + 5) { self.captureAnalyze() }
            } else if self.readinessAttempts < 1200 {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.waitUntilRendered() }
            } else {
                self.fail("Timed out waiting for Streamlit inference to render")
            }
        }
    }

    private func clickTab(_ name: String, completion: @escaping (Bool) -> Void) {
        let quoted = String(data: try! JSONSerialization.data(withJSONObject: name, options: .fragmentsAllowed), encoding: .utf8)!
        let js = "(() => { const name = \(quoted); const tab = Array.from(document.querySelectorAll('[role=tab]')).find(x => x.innerText.trim() === name); if (!tab) return false; tab.click(); return true; })()"
        webView.evaluateJavaScript(js) { value, _ in completion((value as? Bool) == true) }
    }

    private func captureAnalyze() {
        capture("app_analyze_en.png") { self.captureResults() }
    }

    private func captureResults() {
        clickTab("Results") { success in
            guard success else { self.fail("Results tab was not found"); return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                self.capture("app_results_en.png") { self.captureLimitations() }
            }
        }
    }

    private func captureLimitations() {
        clickTab("Limitations") { success in
            guard success else { self.fail("Limitations tab was not found"); return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                self.capture("app_limitations_en.png") { self.captureRobustAnalyze() }
            }
        }
    }

    private func captureRobustAnalyze() {
        clickTab("Analyze") { success in
            guard success else { self.fail("Analyze tab was not found"); return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                let js = "(() => { const box = Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).find(x => x.innerText.includes('TinyCD model')); const combo = box && box.querySelector('[role=combobox]'); if (!combo) return false; combo.click(); setTimeout(() => { const option = Array.from(document.querySelectorAll('[role=option]')).find(x => x.innerText.includes('ROBUST-v2 (blur/order-robust fine-tune)')); if (option) option.click(); }, 500); return true; })()"
                self.webView.evaluateJavaScript(js) { value, error in
                    guard error == nil, (value as? Bool) == true else {
                        self.fail("Could not select ROBUST-v2")
                        return
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 7) {
                        self.capture("app_analyze_en_robust.png") {
                            print("Saved English screenshots to \(self.output.path)")
                            NSApp.terminate(nil)
                        }
                    }
                }
            }
        }
    }

    private func capture(_ name: String, completion: @escaping () -> Void) {
        let configuration = WKSnapshotConfiguration()
        configuration.rect = CGRect(x: 0, y: 0, width: 1440, height: 1800)
        configuration.snapshotWidth = 1440
        webView.takeSnapshot(with: configuration) { [weak self] image, error in
            guard let self else { return }
            guard let image, error == nil,
                  let tiff = image.tiffRepresentation,
                  let bitmap = NSBitmapImageRep(data: tiff),
                  let png = bitmap.representation(using: .png, properties: [:]) else {
                self.fail("Snapshot failed for \(name): \(error?.localizedDescription ?? "unknown error")")
                return
            }
            do {
                try FileManager.default.createDirectory(at: self.output, withIntermediateDirectories: true)
                try png.write(to: self.output.appendingPathComponent(name), options: .atomic)
                print("Saved \(name)")
                DispatchQueue.main.async(execute: completion)
            } catch {
                self.fail("Could not save \(name): \(error.localizedDescription)")
            }
        }
    }

    private func fail(_ message: String) {
        fputs("Screenshot failure: \(message)\n", stderr)
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let driver = ScreenshotDriver(output: output)
DispatchQueue.main.async { driver.start(url: URL(string: "http://127.0.0.1:8501")!) }
app.run()
