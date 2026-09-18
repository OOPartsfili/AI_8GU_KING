# 八股王 iPhone 离线版

SwiftUI + WKWebView 原生应用，最低 iOS 17，适用于 iPhone 16。完整学习内容随包安装，启动不连接题库服务器。

## 使用与安装

Windows 用户看 [安装到 iPhone](安装到iPhone.md)。构建产物 `AI8GUKing-1.5.0-unsigned.ipa` 需先签名，不能在手机 Files 中直接安装。

复习进度通过 WKWebView 的持久本地存储保存。阅读器复制到 App 的 Application Support 固定路径；覆盖升级使用同一数据位置。原生分享面板导出 JSON，系统文件选择器导入 JSON，与电脑版格式兼容。卸载 App 会删除本机数据。

## 从源代码构建

在 macOS + Xcode 16.4 或更高版本执行：

```sh
python3 ios/tools/prepare_assets.py
python3 ios/tools/generate_project.py
open ios/AI8GUKing.xcodeproj
```

用 Xcode 选择自己的 Signing Team 和手机可直接开发安装。项目不依赖 CocoaPods、Swift Package、CDN 或 Node 服务。

Windows 无法运行 Xcode。仓库的 **Actions → Build offline iPhone app** 使用 macOS 执行真实编译、实体 arm64 包校验和 iPhone 16 模拟器测试；成功后下载 `AI8GUKing-iPhone-unsigned` artifact。私有仓库使用账号现有 Actions 配额，构建不会配置 Apple 账号或购买签名服务。

### 离线资源与界面

- 根目录 `开始阅读.html` 是生成输入，Markdown 正文仍是唯一编辑源。
- `tools/prepare_assets.py` 注入手机版布局、原生导入导出桥和禁止网络依赖的 CSP；保留每道题完整数据。
- `AI8GUKing/Resources` 必须作为 folder reference 保持目录结构。
- `OfflineAssetInstaller.swift` 检查内置资源并复制到稳定位置；只有资源版本变化才更新内容文件。
- `OfflineWebView.swift` 在加载 HTML 之前安装网络拦截规则；外部网址仅在用户点击时交给系统浏览器。
- 页面只在需要时解析题目和渲染正文，沿用桌面性能优化。

### 校验

```sh
python3 -m unittest discover -s ios/tests -p 'test_*.py' -v
node ios/tests/check_reader.cjs
```

Node 脚本使用 Playwright 和本机浏览器，可通过 `KNOWLEDGE_BROWSER` 指定浏览器位置。Edge 测试仅覆盖网页层；macOS 的 hosted XCTest 使用 App 真正创建的 WKWebView 验证离线资源、公式、桥接和存储，XCUITest 检查手机界面。

### 包格式

`tools/package_ipa.py` 验证 `CFBundleSupportedPlatforms=iPhoneOS`、arm64 Mach-O 和内置题库 SHA-256，再生成 `Payload/AI8GUKing.app` 的 IPA。模拟器 `.app` 不会被当成手机安装包。包清单标明未签名状态，交给 Windows 个人签名工具处理。

项目不收集 Apple ID、密码、设备 UDID 或签名证书。不要把 `.p12`、provisioning profile、Apple 凭据提交到仓库。
