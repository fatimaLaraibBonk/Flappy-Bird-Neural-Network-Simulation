import numpy as np
import pygame

#Live visualiser for the neural network.
#It only READS from the model (it calls model.calculation, which has no side effects),
#so it never affects training, exploration or the game itself.

INPUT_LABELS = ["Bird Y", "Velocity", "Pillar dist", "Gap offset"]
OUTPUT_LABELS = ["No flap", "Flap"]

_font = None

def _get_font():
    global _font
    if _font is None:
        _font = pygame.font.SysFont(None, 18)
    return _font

def _norm(a):
    #scale a layer's values into -1..1 so brightness is comparable inside a layer
    peak = np.max(np.abs(a))
    return a / peak if peak > 1e-9 else a * 0

def draw_network(surface, model, state, x=10, y=10, width=400, height=260):
    font = _get_font()
    q_values, activations, _ = model.calculation(state)
    layers = [a.flatten() for a in activations] + [q_values.flatten()]
    weights = model.weights

    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (15, 15, 25, 200), (0, 0, width, height), border_radius=8)

    pad_x, pad_top, pad_bottom = 90, 28, 14
    inner_w = width - 2 * pad_x
    inner_h = height - pad_top - pad_bottom
    cols = [pad_x + i * inner_w / (len(layers) - 1) for i in range(len(layers))]
    pos = [
        [(cols[i], pad_top + (j + 1) * inner_h / (len(a) + 1)) for j in range(len(a))]
        for i, a in enumerate(layers)
    ]

    #connections: colour = weight sign, brightness = signal actually flowing (weight * source activation)
    for i, W in enumerate(weights):
        flow = np.abs(W * layers[i][np.newaxis, :])
        peak = flow.max() if flow.max() > 1e-9 else 1
        for j, src in enumerate(pos[i]):
            for k, dst in enumerate(pos[i + 1]):
                s = flow[k][j] / peak
                v = int(25 + 205 * s)
                colour = (30, v, 60) if W[k][j] > 0 else (v, 30, 60)
                pygame.draw.line(panel, colour, src, dst, 1 if s < 0.5 else 2)

    #neurons: brightness = activation, outline highlights the action the network would pick
    chosen = int(np.argmax(layers[-1]))
    for i, layer in enumerate(pos):
        n = _norm(layers[i])
        for j, p in enumerate(layer):
            v = int(40 + 215 * abs(n[j]))
            fill = (v, v, v) if i < len(pos) - 1 else ((60, v, 60) if j == chosen else (v, 60, 60))
            pygame.draw.circle(panel, fill, p, 8)
            outline = (255, 220, 0) if (i == len(pos) - 1 and j == chosen) else (170, 170, 170)
            pygame.draw.circle(panel, outline, p, 8, 1)

    #labels
    for j, p in enumerate(pos[0]):
        t = font.render(INPUT_LABELS[j] if j < len(INPUT_LABELS) else f"in {j}", True, (220, 220, 220))
        panel.blit(t, (p[0] - 12 - t.get_width(), p[1] - t.get_height() // 2))
    for j, p in enumerate(pos[-1]):
        label = OUTPUT_LABELS[j] if j < len(OUTPUT_LABELS) else f"out {j}"
        t = font.render(f"{label} {layers[-1][j]:.2f}", True, (220, 220, 220))
        panel.blit(t, (p[0] + 12, p[1] - t.get_height() // 2))
    title = font.render("Network activity", True, (255, 255, 255))
    panel.blit(title, (10, 7))

    surface.blit(panel, (x, y))
