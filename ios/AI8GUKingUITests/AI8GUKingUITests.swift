import XCTest

final class AI8GUKingUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testOfflineReaderAndPhoneNavigation() throws {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 45))
        let all = app.buttons["全部题目"]
        XCTAssertTrue(all.waitForExistence(timeout: 45), app.debugDescription)
        all.tap()
        XCTAssertTrue(app.staticTexts["341 道题 · 点击题目展开"].waitForExistence(timeout: 15), app.debugDescription)
        app.buttons["Agent"].tap()
        XCTAssertTrue(app.staticTexts["Agent 50 题：从一次调用，到稳定办事"].waitForExistence(timeout: 15), app.debugDescription)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "iPhone16-offline-agent"
        screenshot.lifetime = .keepAlways
        add(screenshot)

        app.buttons["更多"].tap()
        app.buttons["关于离线版"].tap()
        XCTAssertTrue(app.alerts.firstMatch.waitForExistence(timeout: 5))
        XCTAssertTrue(app.alerts.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", "阅读无需网络")).firstMatch.exists)
        app.alerts.buttons["知道了"].tap()

        app.buttons["更多"].tap()
        app.buttons["重新载入题库"].tap()
        XCTAssertTrue(app.buttons["全部题目"].waitForExistence(timeout: 20))
        app.buttons["全部题目"].tap()
        XCTAssertTrue(app.staticTexts["341 道题 · 点击题目展开"].waitForExistence(timeout: 15))
        XCUIDevice.shared.orientation = .landscapeLeft
        let landscape = XCTAttachment(screenshot: app.screenshot())
        landscape.name = "iPhone16-landscape"
        landscape.lifetime = .keepAlways
        add(landscape)
        XCUIDevice.shared.orientation = .portrait
    }
}
