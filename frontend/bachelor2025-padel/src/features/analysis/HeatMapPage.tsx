import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import ParticleBackground from "../../globalComponents/ParticleBackground";
import Animation from "../../globalComponents/Animation";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

type SideLabel = "VENSTRE" | "HØJRE";
type PlayerId = "player_1" | "player_2";

const PLAYER_TO_SIDE: Record<PlayerId, SideLabel> = {
    player_1: "HØJRE",
    player_2: "VENSTRE",
};

const SIDES_ORDER: SideLabel[] = ["VENSTRE", "HØJRE"];

async function fetchHeatmapUrl(matchId: string, playerId: PlayerId): Promise<string> {
    const res = await fetch(`${API_BASE}/heatmaps/matches/${matchId}/players/${playerId}/image`, {
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idToken")}`,
        },
    });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Kunne ikke hente heatmap (${res.status})`);
    }

    const blob = await res.blob();
    return URL.createObjectURL(blob);
}

const HeatmapPage = () => {
    const { videoId } = useParams();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [p1Url, setP1Url] = useState<string | null>(null);
    const [p2Url, setP2Url] = useState<string | null>(null);

    useEffect(() => {
        return () => {
            if (p1Url) URL.revokeObjectURL(p1Url);
            if (p2Url) URL.revokeObjectURL(p2Url);
        };
    }, [p1Url, p2Url]);

    useEffect(() => {
        let cancelled = false;

        const run = async () => {
            if (!videoId) return;
            setLoading(true);
            setError(null);

            try {
                const [u1, u2] = await Promise.all([
                    fetchHeatmapUrl(videoId, "player_1"),
                    fetchHeatmapUrl(videoId, "player_2"),
                ]);

                if (!cancelled) {
                    setP1Url(u1);
                    setP2Url(u2);
                }
            } catch (e: any) {
                if (!cancelled) setError(e?.message ?? "Kunne ikke hente heatmaps");
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        run().then();
        return () => {
            cancelled = true;
        };
    }, [videoId]);

    const cards = useMemo(() => {
        const raw = [
            { playerId: "player_1" as const, side: PLAYER_TO_SIDE.player_1, url: p1Url },
            { playerId: "player_2" as const, side: PLAYER_TO_SIDE.player_2, url: p2Url },
        ];

        return raw.sort((a, b) => SIDES_ORDER.indexOf(a.side) - SIDES_ORDER.indexOf(b.side));
    }, [p1Url, p2Url]);

    return (
        <div className="relative min-h-screen bg-black overflow-hidden">
            <ParticleBackground />

            <Animation>
                <div className="relative z-10 min-h-screen flex flex-col justify-center px-10 py-16">
                    <h1 className="text-5xl font-bold text-center mb-10">HEATMAP</h1>

                    {loading && <p className="text-center text-white/70">Indlæser heatmaps…</p>}
                    {error && <p className="text-center text-red-300 max-w-2xl mx-auto">{error}</p>}

                    {!loading && !error && (
                        <div className="flex flex-col gap-8 max-w-6xl w-full mx-auto">
                            {cards.map((c) => (
                                <div
                                    key={c.playerId}
                                    className="rounded-xl border border-white/10 backdrop-blur-sm overflow-hidden group hover:scale-[1.02] transition"
                                >
                                    <div className="p-5 border-b border-white/10 flex items-center justify-between">
                                        <h2 className="text-2xl font-semibold tracking-wide">{c.side}</h2>
                                    </div>

                                    <div className="relative aspect-video bg-white/5">
                                        {c.url ? (
                                            <>
                                                <img
                                                    src={c.url}
                                                    alt={`Heatmap ${c.playerId}`}
                                                    className="absolute inset-0 w-full h-full object-cover"
                                                />
                                            </>
                                        ) : (
                                            <div className="absolute inset-0 flex items-center justify-center text-white/60">
                                                Ingen billede
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </Animation>
        </div>
    );
};

export default HeatmapPage;
