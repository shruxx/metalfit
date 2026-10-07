// The app icon in colour: the same mark as the menu bar item (Icon.swift), on a macOS tile.
import AppKit

func hex(_ v: UInt32, _ a: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: CGFloat((v >> 16) & 0xff) / 255, green: CGFloat((v >> 8) & 0xff) / 255,
            blue: CGFloat(v & 0xff) / 255, alpha: a)
}

func drawAppIcon(size: CGFloat) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size),
                               bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = size / 1024                          // everything below is laid out on a 1024 canvas

    // macOS app tile: 824 of 1024 with the usual margin, a soft shadow below
    let tile = CGRect(x: 100 * s, y: 100 * s, width: 824 * s, height: 824 * s)
    let tilePath = NSBezierPath(roundedRect: tile, xRadius: 185 * s, yRadius: 185 * s)
    NSGraphicsContext.saveGraphicsState()
    let shadow = NSShadow()
    shadow.shadowColor = hex(0x000000, 0.35)
    shadow.shadowBlurRadius = 24 * s
    shadow.shadowOffset = NSSize(width: 0, height: -10 * s)
    shadow.set()
    hex(0x1A1D23).setFill()
    tilePath.fill()
    NSGraphicsContext.restoreGraphicsState()
    NSGradient(starting: hex(0x323844), ending: hex(0x14161B))!.draw(in: tilePath, angle: -90)

    // the mark, silver chip and coloured layers
    let m = Mark(in: tile.insetBy(dx: 115 * s, dy: 115 * s), pinsPerSide: 3, line: 30 * s)
    let silver = NSGradient(starting: hex(0xE9EDF2), ending: hex(0x9AA3AF))!
    for p in m.pins {
        silver.draw(in: NSBezierPath(roundedRect: p, xRadius: 6 * s, yRadius: 6 * s), angle: -90)
    }
    hex(0x0F1115).setFill()
    m.bodyPath().fill()
    let ring = m.bodyPath()
    ring.lineWidth = m.line
    NSGraphicsContext.saveGraphicsState()
    // stroke with a gradient: draw the stroke as a shape and fill it
    let outline = NSBezierPath(cgPath: ring.cgPath.copy(strokingWithWidth: m.line, lineCap: .butt,
                                                       lineJoin: .round, miterLimit: 10))
    silver.draw(in: outline, angle: -90)
    NSGraphicsContext.restoreGraphicsState()

    // bottom layer deepest blue, top layer brightest cyan: the stack filling the chip
    let colours: [(UInt32, UInt32)] = [(0x3A5BFF, 0x4C7DFF), (0x3F7BFF, 0x4FA0FF),
                                       (0x45A0FF, 0x52C4F5), (0x4CC6F0, 0x6BE8E0)]
    for (b, c) in zip(m.bars, colours) {
        NSGradient(starting: hex(c.0), ending: hex(c.1))!.draw(in: m.barPath(b), angle: 0)
    }

    // a faint highlight on the upper half of the tile, as Apple's own icons have
    NSGraphicsContext.saveGraphicsState()
    tilePath.addClip()
    let gloss = NSBezierPath(rect: CGRect(x: tile.minX, y: tile.midY, width: tile.width, height: tile.height / 2))
    NSGradient(starting: hex(0xFFFFFF, 0.07), ending: hex(0xFFFFFF, 0.0))!.draw(in: gloss, angle: -90)
    NSGraphicsContext.restoreGraphicsState()

    NSGraphicsContext.restoreGraphicsState()
    return rep
}
