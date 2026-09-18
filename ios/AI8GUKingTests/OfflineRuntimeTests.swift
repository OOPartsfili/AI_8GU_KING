import XCTest
import UIKit
import WebKit
@testable import AI8GUKing

/// Runs inside the real application on an isolated simulator. It does not create a
/// replacement web view, mock the bridge, or enable a special application mode.
@MainActor
final class OfflineRuntimeTests: XCTestCase {
    func testRealAppOfflineReadingPersistenceAndNativeExport() async throws {
        let webView = try await waitForApplicationWebView()
        try await waitForReader(in: webView)

        XCTAssertEqual(webView.accessibilityIdentifier, "offlineReader")
        XCTAssertTrue(webView.configuration.websiteDataStore.isPersistent)
        XCTAssertTrue(webView.url?.isFileURL == true)
        let support = try FileManager.default.url(for: .applicationSupportDirectory,
                                                  in: .userDomainMask,
                                                  appropriateFor: nil, create: false)
        XCTAssertEqual(webView.url?.standardizedFileURL.path,
                       support.appendingPathComponent("Web/reader.html").standardizedFileURL.path)

        let initial = try await dictionary("""
        (() => ({
          questions: DB.questions.length,
          uniqueQuestions: new Set(DB.questions.map(q => q.id)).size,
          lazyRecords: document.querySelectorAll('script[id^="data-q-"]').length,
          nativeBridge: !!window.webkit?.messageHandlers?.nativeBridge,
          policy: document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content || '',
          externalResources: performance.getEntriesByType('resource')
            .filter(entry => /^(https?|wss?):/i.test(entry.name)).map(entry => entry.name),
          declaredExternalResources: Array.from(document.querySelectorAll(
            'script[src],img[src],iframe[src],link[rel="stylesheet"][href],audio[src],video[src]'))
            .map(element => element.src || element.href)
            .filter(url => /^(https?|wss?):/i.test(url))
        }))()
        """, in: webView)
        XCTAssertEqual(initial["questions"] as? Int, 341)
        XCTAssertEqual(initial["uniqueQuestions"] as? Int, 341)
        XCTAssertEqual(initial["lazyRecords"] as? Int, 341)
        XCTAssertEqual(initial["nativeBridge"] as? Bool, true)
        XCTAssertTrue((initial["policy"] as? String ?? "").contains("connect-src 'none'"))
        XCTAssertEqual(initial["externalResources"] as? [String], [])
        XCTAssertEqual(initial["declaredExternalResources"] as? [String], [])

        // This simulator belongs to the CI job; no user browser/profile is touched.
        _ = try await evaluate("localStorage.clear(); true", in: webView)
        try await reloadReader(webView)

        let imported = try await dictionary("""
        window.AI8GU.importProgress(JSON.stringify({
          schema: 'llm-study-progress-v1',
          progress: {
            M02: {state: 'done', updated: '2026-09-18T00:00:00Z'},
            UNKNOWN_QUESTION: {state: 'done'},
            M03: {state: '<script>throw new Error("invalid state")</script>'}
          }
        }))
        """, in: webView)
        XCTAssertEqual(imported["ok"] as? Bool, true)
        XCTAssertEqual(imported["count"] as? Int, 1)

        let malformed = try await dictionary("window.AI8GU.importProgress('{broken')", in: webView)
        XCTAssertEqual(malformed["ok"] as? Bool, false)
        let wrongSchema = try await dictionary("window.AI8GU.importProgress('{}')", in: webView)
        XCTAssertEqual(wrongSchema["ok"] as? Bool, false)

        try await reloadReader(webView)
        let saved = try await dictionary("""
        (() => {
          const saved = JSON.parse(localStorage.getItem('llm-interview-study-v1') || '{}');
          return {
            state: saved.M02?.state,
            updated: saved.M02?.updated,
            keys: Object.keys(saved),
            unknown: Object.hasOwn(saved, 'UNKNOWN_QUESTION'),
            invalid: Object.hasOwn(saved, 'M03')
          };
        })()
        """, in: webView)
        XCTAssertEqual(saved["state"] as? String, "done")
        XCTAssertEqual(saved["updated"] as? String, "2026-09-18T00:00:00Z")
        XCTAssertEqual(saved["keys"] as? [String], ["M02"])
        XCTAssertEqual(saved["unknown"] as? Bool, false)
        XCTAssertEqual(saved["invalid"] as? Bool, false)

        // Navigate and expand the same production controls used by the reader.
        _ = try await evaluate("""
        document.getElementById('depth').value = 'full';
        document.getElementById('depth').dispatchEvent(new Event('change'));
        go('q-M02'); true
        """, in: webView)
        try await waitUntil("M02 body and deep formulas did not render", timeout: 15) {
            let ready = try? await self.evaluate("""
            !!document.querySelector('#q-M02 .qbody') &&
            !!document.querySelector('#q-M02 .deep[open] math')
            """, in: webView)
            return ready as? Bool == true
        }
        let question = try await dictionary("""
        (() => {
          const question = document.getElementById('q-M02');
          return {
            textLength: question.innerText.length,
            math: question.querySelectorAll('math').length,
            mathErrors: question.querySelectorAll('merror').length,
            state: question.querySelector('.study').value,
            overflow: document.documentElement.scrollWidth - innerWidth,
            externalResources: performance.getEntriesByType('resource')
              .filter(entry => /^(https?|wss?):/i.test(entry.name)).map(entry => entry.name)
          };
        })()
        """, in: webView)
        XCTAssertGreaterThan(question["textLength"] as? Int ?? 0, 100)
        XCTAssertGreaterThan(question["math"] as? Int ?? 0, 2)
        XCTAssertEqual(question["mathErrors"] as? Int, 0)
        XCTAssertEqual(question["state"] as? String, "done")
        XCTAssertLessThanOrEqual(question["overflow"] as? Int ?? Int.max, 1)
        XCTAssertEqual(question["externalResources"] as? [String], [])

        // Exercise the real JS -> WKScriptMessageHandler -> atomic file export path.
        let previousExports = Set(try exportedFiles().map(\.path))
        let exportAccepted = try await evaluate("window.AI8GU.exportProgress()", in: webView)
        XCTAssertEqual(exportAccepted as? Bool, true)
        var newExport: URL?
        try await waitUntil("Native bridge did not create a progress backup", timeout: 15) {
            newExport = try self.exportedFiles().first { !previousExports.contains($0.path) }
            return newExport != nil
        }
        let exportURL = try XCTUnwrap(newExport)
        let exportData = try Data(contentsOf: exportURL)
        let exportObject = try XCTUnwrap(try JSONSerialization.jsonObject(with: exportData) as? [String: Any])
        XCTAssertEqual(exportObject["schema"] as? String, "llm-study-progress-v1")
        let exportedProgress = try XCTUnwrap(exportObject["progress"] as? [String: Any])
        let exportedM02 = try XCTUnwrap(exportedProgress["M02"] as? [String: Any])
        XCTAssertEqual(exportedM02["state"] as? String, "done")
        XCTAssertEqual(Set(exportedProgress.keys), Set(["M02"]))

        try await waitUntil("Native share sheet was not presented", timeout: 15) {
            self.applicationWindows.contains { window in
                self.containsShareController(window.rootViewController)
            }
        }
        let attachment = XCTAttachment(data: exportData, uniformTypeIdentifier: "public.json")
        attachment.name = "Native-exported-progress.json"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private var applicationWindows: [UIWindow] {
        UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows).sorted { $0.isKeyWindow && !$1.isKeyWindow }
    }

    private func findWebView(in view: UIView) -> WKWebView? {
        if let webView = view as? WKWebView { return webView }
        for child in view.subviews {
            if let result = findWebView(in: child) { return result }
        }
        return nil
    }

    private func waitForApplicationWebView() async throws -> WKWebView {
        var found: WKWebView?
        try await waitUntil("The real application did not create its WKWebView", timeout: 45) {
            found = self.applicationWindows.lazy.compactMap { self.findWebView(in: $0) }.first
            return found != nil
        }
        return try XCTUnwrap(found)
    }

    private func waitForReader(in webView: WKWebView, previousTimeOrigin: String? = nil) async throws {
        try await waitUntil("The bundled reader JavaScript did not become ready", timeout: 45) {
            guard !webView.isLoading else { return false }
            let value = try? await self.evaluate("""
            typeof window.AI8GU?.importProgress === 'function' &&
            typeof DB !== 'undefined' && document.readyState === 'complete'
              ? String(performance.timeOrigin) : null
            """, in: webView)
            guard let origin = value as? String else { return false }
            return previousTimeOrigin == nil || origin != previousTimeOrigin
        }
    }

    private func reloadReader(_ webView: WKWebView) async throws {
        let previous = try await evaluate("String(performance.timeOrigin)", in: webView) as? String
        XCTAssertNotNil(webView.reload(), "The reader must be an actual loaded document")
        try await waitForReader(in: webView, previousTimeOrigin: previous)
    }

    private func evaluate(_ script: String, in webView: WKWebView) async throws -> Any? {
        try await withCheckedThrowingContinuation { continuation in
            webView.evaluateJavaScript(script) { value, error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume(returning: value) }
            }
        }
    }

    private func dictionary(_ script: String, in webView: WKWebView) async throws -> [String: Any] {
        let result = try await evaluate(script, in: webView)
        return try XCTUnwrap(result as? [String: Any], "JavaScript did not return an object")
    }

    private func exportedFiles() throws -> [URL] {
        let manager = FileManager.default
        let directories = try manager.contentsOfDirectory(at: manager.temporaryDirectory,
                                                          includingPropertiesForKeys: nil)
        return directories.filter { $0.lastPathComponent.hasPrefix("AI8GU-Export-") }
            .map { $0.appendingPathComponent("八股王_复习进度.json") }
            .filter { manager.fileExists(atPath: $0.path) }
    }

    private func containsShareController(_ controller: UIViewController?) -> Bool {
        guard let controller else { return false }
        if controller is UIActivityViewController { return true }
        if containsShareController(controller.presentedViewController) { return true }
        return controller.children.contains { containsShareController($0) }
    }

    private func waitUntil(_ failure: String, timeout: TimeInterval,
                           condition: () async throws -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if try await condition() { return }
            try await Task.sleep(nanoseconds: 200_000_000)
        } while Date() < deadline
        throw NSError(domain: "AI8GUKing.OfflineRuntimeTests", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: failure])
    }
}
