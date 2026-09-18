import Foundation
import SwiftUI
import WebKit

struct ReaderNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

struct SharedFile: Identifiable {
    let id = UUID()
    let url: URL
}

@MainActor
final class ReaderModel: ObservableObject {
    @Published var reader: InstalledReader?
    @Published var isLoading = true
    @Published var isReady = false
    @Published var startupFailure: String?
    @Published var showImporter = false
    @Published var isImporting = false
    @Published var notice: ReaderNotice?
    @Published var shareFile: SharedFile?
    weak var webView: WKWebView?
    private var preparing = false
    private var sharedDirectories: [URL] = []
    static let maximumImportBytes = 2_000_000
    static let maximumExportBytes = 10 * 1024 * 1024

    func prepareReader() async {
        guard reader == nil, !preparing else { return }
        preparing = true
        isLoading = true
        startupFailure = nil
        defer { preparing = false }
        do {
            reader = try await Task.detached(priority: .userInitiated) {
                try OfflineAssetInstaller.install()
            }.value
        } catch {
            isLoading = false
            startupFailure = error.localizedDescription
        }
    }

    func reload() {
        guard let reader, let webView else { return }
        isReady = false
        isLoading = true
        webView.loadFileURL(reader.entry, allowingReadAccessTo: reader.directory)
    }

    func exportProgress() {
        guard let webView, isReady else { return }
        webView.callAsyncJavaScript("return window.AI8GU.exportProgress();", arguments: [:], in: nil, in: .page) { [weak self] result in
            if case .failure(let error) = result {
                self?.showError("导出失败", error: error)
            }
        }
    }

    func receiveImport(_ result: Result<URL, Error>) {
        guard !isImporting else { return }
        isImporting = true
        Task { [weak self] in
            guard let self else { return }
            do {
                let url = try result.get()
                let maximumBytes = Self.maximumImportBytes
                // Files/iCloud providers can block while materializing a document.
                // Keep security scope, coordinated I/O and JSON parsing off MainActor.
                let json = try await Task.detached(priority: .userInitiated) {
                    try ProgressFileReader.read(at: url, maximumBytes: maximumBytes)
                }.value
                guard let webView = self.webView, self.isReady else {
                    throw ReaderFailure("题库尚未加载完成，请稍后再导入。")
                }
                webView.callAsyncJavaScript("return await window.AI8GU.importProgress(json);", arguments: ["json": json], in: nil, in: .page) { [weak self] result in
                    self?.isImporting = false
                    switch result {
                    case .success(let value):
                        let response = value as? [String: Any]
                        let ok = response?["ok"] as? Bool ?? false
                        self?.notice = ReaderNotice(title: ok ? "进度已导入" : "导入失败", message: response?["message"] as? String ?? "没有收到有效的导入结果，请重新打开题库后重试。")
                    case .failure(let error):
                        self?.showError("导入失败", error: error)
                    }
                }
            } catch {
                self.isImporting = false
                self.showError("导入失败", error: error)
            }
        }
    }

    func exportFile(filename: String, content: String) {
        do {
            let data = Data(content.utf8)
            guard data.count <= Self.maximumExportBytes else { throw ReaderFailure("导出文件超过 10 MB。") }
            // An untrusted filename must never escape our temporary export directory.
            let lastComponent = (filename as NSString).lastPathComponent
            let safeName = lastComponent.unicodeScalars.filter {
                !CharacterSet.controlCharacters.contains($0) && $0 != "/" && $0 != "\\" && $0 != ":"
            }.map(String.init).joined()
            guard !safeName.isEmpty, safeName != ".", safeName != "..", safeName.utf8.count <= 240 else {
                throw ReaderFailure("无法导出：文件名无效。")
            }
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("AI8GU-Export-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            sharedDirectories.append(directory)
            let destination = directory.appendingPathComponent(safeName)
            try data.write(to: destination, options: .atomic)
            shareFile = SharedFile(url: destination)
        } catch {
            showError("导出失败", error: error)
        }
    }

    func cleanSharedFiles() {
        for directory in sharedDirectories { try? FileManager.default.removeItem(at: directory) }
        sharedDirectories.removeAll()
    }

    func showAbout() {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        notice = ReaderNotice(title: "八股王 · 离线版 \(version)", message: "题库、公式和代码随应用保存在手机中，阅读无需网络。复习进度自动保存在本机，可导出 JSON 备份或与电脑版交换。卸载应用会删除本机进度；卸载前请导出备份。资料来源链接仅在你点击后交给系统浏览器打开。")
    }

    func showError(_ title: String, error: Error) {
        notice = ReaderNotice(title: title, message: error.localizedDescription)
    }
}

private enum ProgressFileReader {
    static func read(at url: URL, maximumBytes: Int) throws -> String {
        let granted = url.startAccessingSecurityScopedResource()
        defer { if granted { url.stopAccessingSecurityScopedResource() } }
        let coordinator = NSFileCoordinator(filePresenter: nil)
        var coordinationError: NSError?
        var readResult: Result<String, Error>?
        coordinator.coordinate(readingItemAt: url, options: [], error: &coordinationError) { readableURL in
            readResult = Result {
                let fileSize = try readableURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                guard fileSize <= maximumBytes else {
                    throw ReaderFailure("进度文件超过 2 MB，请选择八股王导出的 JSON 进度文件。")
                }
                // Bound the actual read as well: file-provider size metadata may be
                // missing or may change between inspection and opening the file.
                let handle = try FileHandle(forReadingFrom: readableURL)
                defer { try? handle.close() }
                var data = Data()
                while data.count <= maximumBytes {
                    let next = try handle.read(upToCount: min(65_536, maximumBytes + 1 - data.count)) ?? Data()
                    if next.isEmpty { break }
                    data.append(next)
                }
                guard data.count <= maximumBytes else {
                    throw ReaderFailure("进度文件超过 2 MB，请选择八股王导出的 JSON 进度文件。")
                }
                guard let json = String(data: data, encoding: .utf8),
                      try JSONSerialization.jsonObject(with: data) is [String: Any] else {
                    throw ReaderFailure("请选择有效的 UTF-8 JSON 进度文件。")
                }
                return json
            }
        }
        if let coordinationError { throw coordinationError }
        guard let readResult else { throw ReaderFailure("无法读取所选进度文件，请下载到本机后重试。") }
        return try readResult.get()
    }
}
