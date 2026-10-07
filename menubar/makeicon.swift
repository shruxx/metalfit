// Writes the app icon as an .iconset folder: `makeicon <out.iconset>`, then iconutil makes the .icns.
// build.sh compiles it with Icon.swift and AppIcon.swift, so app icon and menu bar item are one mark.
import AppKit

@main
enum MakeIcon {
    static func main() throws {
        let args = CommandLine.arguments
        let out = URL(fileURLWithPath: args.count > 1 ? args[1] : "AppIcon.iconset")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        for base in [16, 32, 128, 256, 512] {
            for scale in [1, 2] {
                let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
                let png = drawAppIcon(size: CGFloat(base * scale)).representation(using: .png, properties: [:])!
                try png.write(to: out.appendingPathComponent(name))
            }
        }
    }
}
