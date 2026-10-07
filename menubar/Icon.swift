// metalfit's mark: a chip whose inside is filled exactly by a stack of layers - the model fits whole.
//
// Drawn in code so it is sharp at every size and the repository holds no image files.  The menu bar uses it
// as a template image whose fill shows the state; makeicon.swift draws the same mark in colour for the app.
import AppKit

enum MarkState {
    case off        // no server: dashed outline
    case empty      // server up, nothing loaded
    case loading    // half the layers
    case loaded     // every layer: the model is on the GPU whole

    var layers: Int {
        switch self {
        case .off, .empty: return 0
        case .loading: return 2
        case .loaded: return 4
        }
    }
}

/// The geometry, in a square `r`: the chip body, its pins, and the four layer bars inside.
struct Mark {
    let body: CGRect
    let corner: CGFloat
    let line: CGFloat
    let pins: [CGRect]
    let bars: [CGRect]          // bottom to top

    init(in r: CGRect, pinsPerSide: Int, line: CGFloat) {
        let s = r.width
        let pinLen = s * 0.11
        body = r.insetBy(dx: pinLen + line / 2, dy: pinLen + line / 2)
        corner = s * 0.13
        self.line = line

        // pins on all four sides, evenly spaced along each edge, clear of the corners
        var pins: [CGRect] = []
        let pinW = max(line, s * 0.055)
        for i in 0..<pinsPerSide {
            let t = (CGFloat(i) + 1) / (CGFloat(pinsPerSide) + 1)
            let x = body.minX + body.width * (0.12 + 0.76 * t) - pinW / 2
            let y = body.minY + body.height * (0.12 + 0.76 * t) - pinW / 2
            pins.append(CGRect(x: x, y: r.minY, width: pinW, height: pinLen + line))
            pins.append(CGRect(x: x, y: body.maxY - line / 2, width: pinW, height: pinLen + line))
            pins.append(CGRect(x: r.minX, y: y, width: pinLen + line, height: pinW))
            pins.append(CGRect(x: body.maxX - line / 2, y: y, width: pinLen + line, height: pinW))
        }
        self.pins = pins

        // four bars that fill the inside exactly: the gap to the wall equals the gap between bars
        let inner = body.insetBy(dx: line / 2 + s * 0.06, dy: line / 2 + s * 0.06)
        let gap = s * 0.045
        let h = (inner.height - 3 * gap) / 4
        bars = (0..<4).map { i in
            CGRect(x: inner.minX, y: inner.minY + CGFloat(i) * (h + gap), width: inner.width, height: h)
        }
    }

    func bodyPath() -> NSBezierPath {
        NSBezierPath(roundedRect: body, xRadius: corner, yRadius: corner)
    }

    func barPath(_ b: CGRect) -> NSBezierPath {
        let rad = b.height * 0.3
        return NSBezierPath(roundedRect: b, xRadius: rad, yRadius: rad)
    }
}

/// The menu bar image: black on clear, marked as a template so macOS tints it for light, dark and highlight.
func menuBarImage(_ state: MarkState) -> NSImage {
    let size = NSSize(width: 18, height: 18)
    let img = NSImage(size: size, flipped: false) { rect in
        let m = Mark(in: rect.insetBy(dx: 0.5, dy: 0.5), pinsPerSide: 2, line: 1.4)
        NSColor.black.setStroke()
        NSColor.black.setFill()
        let body = m.bodyPath()
        body.lineWidth = m.line
        if state == .off { body.setLineDash([2.0, 1.6], count: 2, phase: 0) }
        body.stroke()
        if state != .off { m.pins.forEach { NSBezierPath(rect: $0).fill() } }
        for b in m.bars.prefix(state.layers) { m.barPath(b).fill() }
        return true
    }
    img.isTemplate = true
    img.accessibilityDescription = "metalfit"
    return img
}
