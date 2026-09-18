# 八股王 iPhone 版 v1.5.0

发布日期：2026-09-18。适用于实体 iPhone，最低 iOS 17；已在 iPhone 16 上安装并完成基本真机验证。

## 下载与使用

[GitHub v1.5.0 下载页](https://github.com/OOPartsfili/AI_8GU_KING/releases/tag/v1.5.0) 提供：

- `AI8GUKing-1.5.0-unsigned.ipa`：实体 iPhone ARM64 未签名安装包。
- `AI8GUKing-1.5.0-iPhone.zip`：安装包、中文安装说明、验收记录和包清单。
- `SHA256SUMS.txt`：IPA 与完整交付 ZIP 的 SHA-256 校验值。

按照 [Windows 安装说明](../ios/安装到iPhone.md)，使用自己的 Apple 账户在 Sideloadly 中签名安装。免费个人签名通常 7 天到期；续签时使用相同账户和应用标识覆盖安装，先导出进度备份。

## 本版功能

- 内置 21 个知识章节、341 道题和 135 条来源；首次启动无需下载题库。
- 公式、代码、Agent 教程、Transformer 手写教程及交互演示均随包保存。
- 复习进度保存在手机本地，支持 JSON 导入与导出，与电脑版格式兼容。
- 适配手机横竖屏，沿用题目按需解析与渲染的加载优化。
- 来源网页在用户点击后交给系统浏览器打开，需要联网；阅读内置内容无需联网。

## 验证证据

| 范围 | 结果与来源 |
|---|---|
| 实体手机包 | 已校验为 iPhoneOS / ARM64，最低 iOS 17；内置 341 道题的元数据与完整正文，内容哈希一致。 |
| 编译与模拟器 | [实际构建与测试通过](https://github.com/OOPartsfili/AI_8GU_KING/actions/runs/35309434734)，包括原生 WKWebView 加载、公式、进度持久化、导入校验和原生导出。 |
| 界面与横屏 | [后续界面测试及完整横屏截图通过](https://github.com/OOPartsfili/AI_8GU_KING/actions/runs/35310917512)。 |
| Windows 安装 | Sideloadly 通过 USB 安装到 iPhone，当前安装结果实际检查为 `100%`、`Done.`。 |
| iPhone 16 基本验收 | 用户于 2026-09-18 反馈“真机验证基本没问题”，并确认上传本版。属于用户实测反馈，不代表穷举所有题目和设备。 |

## 安装包来源

- IPA 文件大小：499,317 字节。
- IPA SHA-256：`7b6c8ef2d8ce880784b87c9d5a5fcce2b6641499eeba38a2bde81d431531b27d`。
- IPA 构建提交：`8f5323c36140e4e63517f4f937c5492964649d55`。
- 后续验证提交：`500d99e79bce2b2850ed5951b9982f8e2c3880bf`，修正测试截图采集；应用源码与交付 IPA 内容未变。
- 本次发布补充下载入口与验收记录，复用用户已安装验证的同一 IPA。

发行文件不包含 Apple 账户、密码、设备标识或个人签名证书。安装包在每位使用者的本机签名，不能通过手机“文件”直接点开安装。
