# 八股王：在 Windows 上安装到 iPhone 16

## 先看交付的是什么

`AI8GUKing-1.5.0-unsigned.ipa` 是为实体 iPhone 编译的 **待签名安装包**，支持 iOS 17 及以上。完整题库、公式、代码和交互演示都已内置；首次打开也不需要下载题库。

iPhone 会验证应用签名，不能把这个 IPA 传到手机的“文件”里直接点开安装。你可以在 Windows 上用自己的 Apple ID 为它签名，再通过数据线安装。不需要 Mac，也不用购买开发者账号才能进行个人安装。

**免费账号的签名有效期通常为 7 天。** 到期前需要在电脑上重新签名或启用工具的自动续签。阅读和复习是离线的，安装及续签需要联网。若想长期免续签分发，需要另行选择开发者计划/TestFlight/App Store 路线；本包没有声称实现免签或永久安装。

## 第一次安装

1. 在 Windows 访问 [Sideloadly 官网](https://sideloadly.io/)，下载对应 Windows 版本。按官网当前提示安装 Apple 官方 iTunes/iCloud 组件，避免从不明下载站获取安装器。
2. 用数据线连接 iPhone 16，解锁手机；出现“信任此电脑”时在手机上确认。
3. 启动 Sideloadly，在设备列表选中这台 iPhone，把 `AI8GUKing-1.5.0-unsigned.ipa` 拖入窗口。
4. 在工具中输入你自己的 Apple ID，点击 **Start**。Apple ID 密码和双重验证码由你在自己的电脑/手机上输入，不要发到聊天里，也不要写入代码仓库。
5. 按提示完成签名和安装。首次运行若提示未受信任，在 iPhone **设置 → 通用 → VPN 与设备管理** 中找到对应开发者账号并信任。
6. 若提示开发者模式，在 **设置 → 隐私与安全 → 开发者模式** 中开启，按提示重启并确认。此选项可能在首次配对/尝试安装开发应用之后才出现。
7. 打开桌面的“八股王”。看到首页、搜索和章节目录即完成第一次启动。

实际选项名称可能随 iOS/工具版本变化，以 Apple 和工具官网提示为准。

## 安装后的三个确认

- **离线题库：** 开启飞行模式后彻底退出 App，再打开，搜索 `M02` 并展开公式；再打开 Agent 和 Transformer 可视化。
- **进度保存：** 将一道题改为“已掌握”，退出再打开，确认状态仍保留。
- **进度备份：** 右上角“更多 → 导出复习进度”，选择“存储到文件”。电脑版导出的同格式 JSON 也可通过“导入复习进度”导入。

## 续签和更新

- 免费个人签名一般 7 天到期；Sideloadly 可设置后台自动续签，但电脑必须运行，手机也需要通过数据线或相同 Wi-Fi 被检测到。
- 持续使用同一 Apple ID、同一应用标识重签并覆盖安装。应用默认标识是 `com.oopartsfili.ai8guking`；请保留工具对该 App 使用的标识。
- **不要为续签先卸载 App。** 卸载会删除本机进度；升级或改换签名方式前先导出备份。
- 来源链接只有在你点击时才会交给系统浏览器打开，打开论文网页需要网络。阅读题库和操作内置演示无需网络。

## 查证来源

- [Apple：App 代码签名与验证](https://support.apple.com/en-ca/guide/security/sec7c917bf14/web)
- [Apple：Personal Team 限制和 7 天有效期](https://developer.apple.com/help/account/basics/about-your-developer-account)
- [Apple：开启 Developer Mode](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device)
- [Sideloadly 官方下载与说明](https://sideloadly.io/)
- [Sideloadly FAQ](https://sideloadly.io/faq)

本说明不要求提供账号给项目维护者。签名由你的本机工具与 Apple 完成。
