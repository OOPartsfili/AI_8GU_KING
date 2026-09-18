import Foundation
import CryptoKit

struct InstalledReader: Sendable {
    let directory: URL
    var entry: URL { directory.appendingPathComponent("reader.html") }
}

enum OfflineAssetInstaller {
    private struct Asset {
        let relativePath: String
        let contents: Data
    }

    static func install() throws -> InstalledReader {
        guard let source = Bundle.main.resourceURL?.appendingPathComponent("OfflineContent", isDirectory: true),
              FileManager.default.fileExists(atPath: source.appendingPathComponent("reader.html").path) else {
            throw ReaderFailure("安装包中缺少离线题库，请重新安装完整版本。")
        }
        let manager = FileManager.default
        guard let enumerator = manager.enumerator(at: source, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) else {
            throw ReaderFailure("无法读取安装包中的题库文件。")
        }
        let files = enumerator.compactMap { $0 as? URL }.sorted { $0.path < $1.path }
        var assets: [Asset] = []
        var digest = SHA256()
        for url in files {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let relative = String(url.path.dropFirst(source.path.count + 1))
            let data = try Data(contentsOf: url)
            digest.update(data: Data(relative.utf8))
            digest.update(data: Data([0]))
            digest.update(data: data)
            assets.append(Asset(relativePath: relative, contents: data))
        }
        let revision = digest.finalize().map { String(format: "%02x", $0) }.joined()
        let support = try manager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        var destination = support.appendingPathComponent("Web", isDirectory: true)
        try manager.createDirectory(at: destination, withIntermediateDirectories: true)
        // These files can always be restored from the application bundle. Progress is
        // in WKWebsiteDataStore.default(), not in this replaceable content directory.
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try destination.setResourceValues(resourceValues)
        let marker = destination.appendingPathComponent(".asset-revision")
        let previousRevision = try? String(contentsOf: marker, encoding: .utf8)
        if previousRevision != revision || !manager.fileExists(atPath: destination.appendingPathComponent("reader.html").path) {
            for asset in assets {
                let target = destination.appendingPathComponent(asset.relativePath)
                try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                // Each replacement is atomic. Write the revision last so an interrupted
                // upgrade repairs itself on the next launch without touching progress.
                try asset.contents.write(to: target, options: .atomic)
            }
            try Data(revision.utf8).write(to: marker, options: .atomic)
        }
        // Never load from the versioned application bundle: using a stable data-container
        // URL prevents ordinary app upgrades from changing the reader's storage origin.
        return InstalledReader(directory: destination)
    }
}

struct ReaderFailure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
