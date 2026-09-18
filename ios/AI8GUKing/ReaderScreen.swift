import SwiftUI
import UniformTypeIdentifiers
import UIKit

struct ReaderScreen: View {
    @StateObject private var model = ReaderModel()

    var body: some View {
        NavigationStack {
            ZStack {
                Color(uiColor: .systemBackground)
                if let reader = model.reader {
                    OfflineWebView(model: model, reader: reader)
                } else if let failure = model.startupFailure {
                    ContentUnavailableView {
                        Label("题库打开失败", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(failure)
                    } actions: {
                        Button("重新打开") { Task { await model.prepareReader() } }
                            .buttonStyle(.borderedProminent)
                    }
                }
                if model.isLoading || model.isImporting {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text(model.isImporting ? "正在导入复习进度…" : "正在打开离线题库…").font(.callout)
                    }
                    .padding(24)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
                    .accessibilityElement(children: .combine)
                }
            }
            // SwiftUI keeps the reader below the status/navigation bars and above the
            // home indicator in both orientations; the HTML does not add a second inset.
            .navigationTitle("八股王")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Label("离线", systemImage: "checkmark.shield")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("题库离线保存在手机中")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("导出复习进度", systemImage: "square.and.arrow.up") {
                            model.exportProgress()
                        }
                        .disabled(!model.isReady || model.isImporting)
                        Button("导入复习进度", systemImage: "square.and.arrow.down") {
                            model.showImporter = true
                        }
                        .disabled(!model.isReady || model.isImporting)
                        Divider()
                        Button("重新载入题库", systemImage: "arrow.clockwise") {
                            model.reload()
                        }
                        .disabled(model.reader == nil || model.isImporting)
                        Button("关于离线版", systemImage: "info.circle") {
                            model.showAbout()
                        }
                    } label: {
                        Label("更多", systemImage: "ellipsis.circle")
                    }
                }
            }
            .fileImporter(isPresented: $model.showImporter, allowedContentTypes: [.json]) { result in
                model.receiveImport(result)
            }
            .sheet(item: $model.shareFile, onDismiss: model.cleanSharedFiles) { item in
                ShareFileSheet(url: item.url)
                    .presentationDetents([.medium, .large])
            }
            .alert(item: $model.notice) { item in
                Alert(title: Text(item.title), message: Text(item.message), dismissButton: .default(Text("知道了")))
            }
            .task { await model.prepareReader() }
        }
        .tint(Color(red: 0.18, green: 0.38, blue: 0.84))
    }
}

private struct ShareFileSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
