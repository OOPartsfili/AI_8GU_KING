import SwiftUI
import WebKit
import UIKit

struct OfflineWebView: UIViewRepresentable {
    @ObservedObject var model: ReaderModel
    let reader: InstalledReader

    func makeCoordinator() -> Coordinator { Coordinator(model: model, reader: reader) }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.userContentController.add(context.coordinator, name: "nativeBridge")
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.accessibilityIdentifier = "offlineReader"
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        webView.scrollView.backgroundColor = .systemBackground
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.allowsBackForwardNavigationGestures = false
        model.webView = webView
        // Wait for the network deny rule before loading any HTML. This also covers
        // subresources, fetch/XHR, images and fonts; navigationDelegate alone does not.
        let rules = #"[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},{"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}}]"#
        WKContentRuleListStore.default().compileContentRuleList(forIdentifier: "AI8GUOfflineOnly-v1", encodedContentRuleList: rules) { [weak webView, weak model] ruleList, error in
            guard let webView, let model else { return }
            if let ruleList {
                webView.configuration.userContentController.add(ruleList)
                webView.loadFileURL(reader.entry, allowingReadAccessTo: reader.directory)
            } else {
                model.isLoading = false
                model.startupFailure = (error ?? ReaderFailure("离线资源规则加载失败，请重新启动应用。")).localizedDescription
                model.reader = nil
                model.webView = nil
            }
        }
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.stopLoading()
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "nativeBridge")
        webView.navigationDelegate = nil
        webView.uiDelegate = nil
    }

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        let model: ReaderModel
        let reader: InstalledReader

        init(model: ReaderModel, reader: InstalledReader) {
            self.model = model
            self.reader = reader
        }

        private func isLocalReaderURL(_ url: URL?) -> Bool {
            guard let url, url.isFileURL else { return false }
            let path = url.standardizedFileURL.resolvingSymlinksInPath().path
            let allowed = reader.directory.standardizedFileURL.resolvingSymlinksInPath().path
            return path.hasPrefix(allowed + "/")
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == "nativeBridge", message.frameInfo.isMainFrame,
                  isLocalReaderURL(message.frameInfo.request.url),
                  let payload = message.body as? [String: Any],
                  let action = payload["action"] as? String else { return }
            switch action {
            case "export":
                guard let filename = payload["filename"] as? String,
                      let content = payload["content"] as? String else { return }
                model.exportFile(filename: filename, content: content)
            case "importProgress":
                model.showImporter = true
            default:
                break
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let url = navigationAction.request.url else { decisionHandler(.cancel); return }
            let scheme = url.scheme?.lowercased()
            if scheme == "http" || scheme == "https" {
                if navigationAction.navigationType == .linkActivated {
                    UIApplication.shared.open(url, options: [:], completionHandler: nil)
                }
                decisionHandler(.cancel)
                return
            }
            if isLocalReaderURL(url) {
                decisionHandler(.allow)
            } else {
                // Downloads use the text bridge; unsupported schemes and remote
                // frames cannot replace the trusted reader or access native actions.
                decisionHandler(.cancel)
            }
        }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
            // The navigation delegate already opens explicit external links. Returning
            // nil prevents target=_blank from creating an unguarded embedded browser.
            nil
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            model.isLoading = false
            model.isReady = true
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { report(error) }
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { report(error) }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            model.isReady = false
            model.isLoading = true
            webView.loadFileURL(reader.entry, allowingReadAccessTo: reader.directory)
        }

        private func report(_ error: Error) {
            if (error as NSError).code == NSURLErrorCancelled { return }
            model.isLoading = false
            model.isReady = false
            model.showError("题库加载失败", error: error)
        }
    }
}
