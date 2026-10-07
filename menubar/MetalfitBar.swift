// metalfit in the menu bar: which model is loaded, and a menu to switch or unload.
//
// It talks to `metalfit serve` over its own API (GET /api/status, POST /api/load and /api/unload) and polls
// every 2 s, so it holds no state of its own and works whether the server was started before or after it.
// It can also start the server and stop it: the command comes from `defaults write io.github.shruxx.metalfit.bar
// command -array ...`, else from what build.sh found and wrote into Info.plist, else from the Python, metalfit
// and llama-server a release build carries in Resources.  Stopping sends SIGTERM to whatever listens on the
// port, which metalfit answers by unloading the model, so llama-server never stays behind holding wired memory.
// Build with menubar/build.sh (release/build.sh for the self-contained app); METALFIT_PORT picks another port.
import AppKit
import ServiceManagement

let port = ProcessInfo.processInfo.environment["METALFIT_PORT"] ?? "8099"
let base = URL(string: "http://127.0.0.1:\(port)")!
let home = FileManager.default.homeDirectoryForCurrentUser
let logURL = home.appendingPathComponent("Library/Logs/metalfit.log")
let defaults = UserDefaults.standard

/// `metalfit serve ...` as an argument list: `defaults` first, then what build.sh baked in, then what the app
/// carries.  The bundled one serves `defaults write ... models <folder>`, else ~/models, made if missing.
func serverCommand() -> [String] {
    if let c = defaults.stringArray(forKey: "command"), !c.isEmpty { return c }
    if let c = Bundle.main.object(forInfoDictionaryKey: "MetalfitCommand") as? [String], !c.isEmpty { return c }
    guard let res = Bundle.main.resourceURL else { return [] }
    let python = res.appendingPathComponent("python/bin/python3").path
    let llama = res.appendingPathComponent("llama/llama-server").path
    guard FileManager.default.isExecutableFile(atPath: python) else { return [] }
    let models = defaults.string(forKey: "models") ?? home.appendingPathComponent("models").path
    try? FileManager.default.createDirectory(atPath: models, withIntermediateDirectories: true)
    return [python, "-m", "metalfit", "serve", "--models", models, "--llama-server", llama]
        + (defaults.stringArray(forKey: "args") ?? [])
}

/// The process listening on our port, found with lsof - so a server started by hand can be stopped too.
func listenerPID() -> pid_t? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
    p.arguments = ["-nP", "-t", "-iTCP:\(port)", "-sTCP:LISTEN"]
    let out = Pipe()
    p.standardOutput = out
    p.standardError = FileHandle.nullDevice
    guard (try? p.run()) != nil else { return nil }
    p.waitUntilExit()
    let s = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return s.split(separator: "\n").first.flatMap { pid_t($0.trimmingCharacters(in: .whitespaces)) }
}

struct Loaded: Decodable, Equatable {
    let name: String
    let path: String
    let n_ctx: Int
    let n_gpu_layers: Int
    let whole: Bool
}

/// What metalfit measured from real replies on this Mac; nil fields were not measured yet.
struct Speed: Decodable, Equatable {
    let gen: Double?
    let gen_long: Double?
    let prompt: Double?
    let replies: Int
}

struct Entry: Decodable, Equatable {
    let name: String
    let path: String
    let file_gb: Double
    let advice: String
    let speed: Speed?
}

struct Status: Decodable {
    let models_dir: String?
    let busy: String
    let keep_alive_s: Double
    let idle_s: Int?
    let loaded: Loaded?
    let models: [Entry]
}

/// "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" -> "Qwen3.6-35B-A3B-UD-Q4_K_XL"; the title is cut at 28 characters so a
/// long name does not push other menu bar items away, the menu shows it whole.
func stem(_ name: String) -> String {
    name.hasSuffix(".gguf") ? String(name.dropLast(5)) : name
}

func short(_ name: String, _ max: Int = 28) -> String {
    let s = stem(name)
    return s.count <= max ? s : String(s.prefix(max - 1)) + "…"
}

final class App: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    var status: Status?
    var reachable = false
    var timer: Timer?
    var server: Process?          // the server this app started, if it did
    var serverState = ""          // "starting" / "stopping" while that is under way, else ""
    var lastError = ""
    var menuOpen = false          // an open menu is rebuilt on every poll, so it never shows a stale list

    func applicationDidFinishLaunching(_ note: Notification) {
        item.button?.image = menuBarImage(.off)
        item.button?.imagePosition = .imageLeading
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.refresh() }
        if defaults.bool(forKey: "autostart") {
            // give the first status poll a moment, so a server that is already up is not started twice
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { if !self.reachable { self.startServer() } }
        }
    }

    // ---- starting and stopping the server
    func startServer() {
        let cmd = serverCommand()
        guard let exe = cmd.first, FileManager.default.isExecutableFile(atPath: exe) else {
            lastError = cmd.isEmpty ? "no server command: run build.sh again, or set it with defaults write"
                                    : "not executable: \(cmd[0])"
            return
        }
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        guard let log = try? FileHandle(forWritingTo: logURL) else { lastError = "cannot write \(logURL.path)"; return }
        log.seekToEndOfFile()
        log.write("\n[metalfit-bar] \(Date()) starting: \(cmd.joined(separator: " "))\n".data(using: .utf8)!)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = Array(cmd.dropFirst())
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"         // the log should show lines as they happen
        env["PYTHONDONTWRITEBYTECODE"] = "1"  // a __pycache__ written into the app would break its signature
        p.environment = env
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { proc in
            DispatchQueue.main.async {
                if self.serverState == "starting" {
                    self.lastError = "the server exited (\(proc.terminationStatus)) - see the log"
                }
                self.server = nil
                self.serverState = ""
                self.refresh()
            }
        }
        do {
            try p.run()
            server = p
            serverState = "starting"
            lastError = ""
        } catch {
            lastError = "could not start: \(error.localizedDescription)"
        }
        updateTitle()
    }

    func stopServer() {
        if let p = server, p.isRunning {
            p.terminate()
        } else if let pid = listenerPID() {
            kill(pid, SIGTERM)
        } else {
            return
        }
        serverState = "stopping"
        lastError = ""
        updateTitle()
    }

    // ---- talking to metalfit
    func refresh() {
        var req = URLRequest(url: base.appendingPathComponent("api/status"))
        req.timeoutInterval = 3
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let s = data.flatMap { try? JSONDecoder().decode(Status.self, from: $0) }
            DispatchQueue.main.async {
                self.status = s
                self.reachable = s != nil
                if (self.serverState == "starting" && self.reachable) ||
                   (self.serverState == "stopping" && !self.reachable && self.server == nil) {
                    self.serverState = ""
                }
                self.updateTitle()
                if self.menuOpen, let menu = self.item.menu { self.menuNeedsUpdate(menu) }
            }
        }.resume()
    }

    func menuWillOpen(_ menu: NSMenu) {
        menuOpen = true
        refresh()                 // the list as it is now, not as it was up to 2 s ago
    }

    func menuDidClose(_ menu: NSMenu) { menuOpen = false }

    func post(_ path: String, _ body: [String: Any] = [:]) {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 600                  // a load answers once the model is warm
        URLSession.shared.dataTask(with: req) { _, _, _ in
            DispatchQueue.main.async { self.refresh() }
        }.resume()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.refresh() }   // show "busy" at once
    }

    // ---- what the menu bar shows: the mark says the state, text only for the loaded model's name
    func updateTitle() {
        guard let button = item.button else { return }
        let state: MarkState
        if !serverState.isEmpty || (status.map { !$0.busy.isEmpty } ?? false) {
            state = .loading
        } else if !reachable {
            state = .off
        } else if status?.loaded != nil {
            state = .loaded
        } else {
            state = .empty
        }
        button.image = menuBarImage(state)
        button.title = state == .loaded ? " " + short(status!.loaded!.name) : ""
        button.appearsDisabled = state == .off
        button.toolTip = status?.loaded.map { stem($0.name) } ?? (reachable ? "metalfit: nothing loaded"
                                                                         : "metalfit is not running")
    }

    // the menu is built when it opens, so it is never stale
    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        if !lastError.isEmpty { menu.addItem(info("⚠︎ " + lastError)) }
        guard reachable, let s = status else {
            if serverState == "starting" {
                menu.addItem(info("starting the server…"))
            } else if serverState == "stopping" {
                menu.addItem(info("stopping the server…"))
            } else {
                menu.addItem(info("metalfit is not running on port \(port)"))
                menu.addItem(action("Start server", #selector(start), key: "s"))
            }
            footer(menu)
            return
        }
        if !s.busy.isEmpty {
            menu.addItem(info(s.busy))
        } else if let l = s.loaded {
            menu.addItem(info(stem(l.name)))
            menu.addItem(info("-c \(l.n_ctx), \(l.n_gpu_layers) layers on the GPU" + (l.whole ? " (whole)" : "")))
            if let idle = s.idle_s, s.keep_alive_s > 0 {
                let left = max(Int(s.keep_alive_s) - idle, 0)
                menu.addItem(info("idle \(idle / 60) min, unloads in \(left / 60) min \(left % 60) s"))
            }
        } else {
            menu.addItem(info("nothing loaded"))
        }
        menu.addItem(.separator())

        for m in s.models {
            var title = stem(m.name) + String(format: "  %.1f GB", m.file_gb)
            if let g = m.speed?.gen ?? m.speed?.gen_long { title += String(format: " · %.0f tok/s", g) }
            let mi = action(title, #selector(load(_:)))
            mi.representedObject = m.path
            var tip = m.advice
            if let sp = m.speed {
                if let g = sp.gen { tip += String(format: "\nwriting: %.1f tok/s", g) }
                if let g = sp.gen_long { tip += String(format: "\nwriting at long context: %.1f tok/s", g) }
                if let p = sp.prompt { tip += String(format: "\nreading prompts: %.0f tok/s", p) }
                tip += "\nmeasured from \(sp.replies) replies on this Mac"
            } else {
                tip += "\nspeed: not measured on this Mac yet"
            }
            mi.toolTip = tip
            mi.state = s.loaded?.path == m.path ? .on : .off
            mi.isEnabled = s.busy.isEmpty
            menu.addItem(mi)
        }
        menu.addItem(.separator())
        let un = action("Unload", #selector(unload), key: "u")
        un.isEnabled = s.loaded != nil && s.busy.isEmpty
        menu.addItem(un)
        menu.addItem(action("Open metalfit page", #selector(openPage), key: "o"))
        if let dir = s.models_dir {
            let mi = action("Open models folder", #selector(openFolder(_:)))
            mi.representedObject = dir
            mi.toolTip = dir + " - .gguf files put here appear in this menu within 2 s"
            menu.addItem(mi)
        }
        menu.addItem(action("Stop server", #selector(stop)))
        footer(menu)
    }

    func footer(_ menu: NSMenu) {
        menu.addItem(.separator())
        let login = action("Open at login", #selector(toggleLogin))
        login.state = SMAppService.mainApp.status == .enabled ? .on : .off
        menu.addItem(login)
        let auto = action("Start server when this app opens", #selector(toggleAutostart))
        auto.state = defaults.bool(forKey: "autostart") ? .on : .off
        menu.addItem(auto)
        menu.addItem(action("Open server log", #selector(openLog), key: "l"))
        menu.addItem(.separator())
        menu.addItem(action("Quit menu bar item", #selector(quit), key: "q"))
    }

    func info(_ text: String) -> NSMenuItem {
        let mi = NSMenuItem(title: text, action: nil, keyEquivalent: "")
        mi.isEnabled = false
        return mi
    }

    func action(_ title: String, _ sel: Selector, key: String = "") -> NSMenuItem {
        let mi = NSMenuItem(title: title, action: sel, keyEquivalent: key)
        mi.target = self
        return mi
    }

    @objc func load(_ sender: NSMenuItem) {
        guard let path = sender.representedObject as? String else { return }
        post("api/load", ["path": path])
    }

    @objc func unload() { post("api/unload") }
    @objc func start() { startServer() }
    @objc func stop() { stopServer() }
    @objc func toggleAutostart() { defaults.set(!defaults.bool(forKey: "autostart"), forKey: "autostart") }
    @objc func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled {
                try SMAppService.mainApp.unregister()
            } else {
                try SMAppService.mainApp.register()
            }
            lastError = ""
        } catch {
            lastError = "login item: \(error.localizedDescription)"
        }
        if SMAppService.mainApp.status == .requiresApproval {
            SMAppService.openSystemSettingsLoginItems()     // macOS wants the user to allow it there
        }
    }
    @objc func openLog() {
        if FileManager.default.fileExists(atPath: logURL.path) { NSWorkspace.shared.open(logURL) }
    }
    @objc func openPage() { NSWorkspace.shared.open(base) }
    @objc func openFolder(_ sender: NSMenuItem) {
        if let dir = sender.representedObject as? String { NSWorkspace.shared.open(URL(fileURLWithPath: dir)) }
    }
    @objc func quit() { NSApp.terminate(nil) }
}

@main
enum Main {
    static func main() {
        let app = NSApplication.shared
        let delegate = App()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)          // menu bar only, no Dock icon
        app.run()
    }
}
