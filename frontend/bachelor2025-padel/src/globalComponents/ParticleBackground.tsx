// ParticleBackground.tsx
import { useCallback } from "react";
import Particles from "react-tsparticles";
import type { Engine } from "tsparticles-engine";
import { loadSlim } from "tsparticles-slim";

export default function ParticleBackground() {
    const init = useCallback(async (engine: Engine) => {
        await loadSlim(engine);
    }, []);

    return (
        <Particles
            id="tsparticles"
            init={init}
            className="absolute inset-0 z-0 pointer-events-none"
            options={{
                background: { color: "transparent" },
                fpsLimit: 60,
                detectRetina: true,
                particles: {
                    number: { value: 120, density: { enable: true, area: 800 } },
                    size: { value: 2, random: true },
                    move: { enable: true, speed: 0.6, outModes: { default: "out" } },
                    color: { value: "#ffffff" },
                    opacity: { value: 1, random: true },
                    links: { enable: false }
                },
                interactivity: {
                    events: { onHover: { enable: false }, onClick: { enable: false }, resize: true },
                },
            }}
        />
    );
}
