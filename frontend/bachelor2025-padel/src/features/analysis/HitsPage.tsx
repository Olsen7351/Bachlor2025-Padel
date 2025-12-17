import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import StatBar from "./StatsBar.tsx";
import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";
import {apiFetch} from "../../utils/apiFetch.ts";

interface PlayerStats {
    player_identifier: string;
    total_hits: number;
    overhead_hits: number;
    lob: number;
    serve: number;
    groundstrokes: number;
}

interface MatchOverview {
    match_id: number;
    analysis_id: number;
    player_statistics: PlayerStats[];
    created_at: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL;

const HitsPage = () => {
    const { videoId } = useParams();
    const [overview, setOverview] = useState<MatchOverview | null>(null);
    const [loading, setLoading] = useState(true);

    const PLAYER_LABELS: Record<string, "SPILLER 1" | "SPILLER 2"> = {
        player_1: "SPILLER 1",
        player_2: "SPILLER 2",
    };



    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await apiFetch(`${API_BASE}/matches/${videoId}/overview`);

                if (!res.ok) {
                    throw new Error(await res.text());
                }

                const json = await res.json();
                setOverview(json);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };

        fetchData().then();
    }, [videoId]);

    if (loading) return <p className="text-center text-white">Indlæser…</p>;
    if (!overview) return <p className="text-center text-red-400">Ingen data</p>;

    const sortedPlayers = [...overview.player_statistics].sort((a, b) => {
        const la = PLAYER_LABELS[a.player_identifier] ?? a.player_identifier;
        const lb = PLAYER_LABELS[b.player_identifier] ?? b.player_identifier;
        return la.localeCompare(lb);
    });


    return (
        <>
            <ParticleBackground />
            <Animation>

                <div className="min-h-screen overflow-hidden flex flex-col justify-center px-10">

                <h1 className="text-5xl font-bold text-center mb-10">
                    SLAG
                </h1>

            <div className="grid grid-cols-2 gap-8 max-w-5xl w-full mx-auto">
                {sortedPlayers.map((player) => {
                    const max = player.total_hits || 1;
                    const label = PLAYER_LABELS[player.player_identifier] ?? player.player_identifier;

                    return (
                        <div
                            key={player.player_identifier}
                            className="rounded-xl border border-white/10 backdrop-blur-sm p-6 hover:scale-[1.02] transition"
                        >
                            <h2 className="text-2xl font-semibold mb-6 text-center tracking-wide">
                                {label}
                            </h2>

                            <div className="space-y-4">
                                <StatBar label="Total slag" value={player.total_hits} max={max} />
                                <StatBar label="Overhead" value={player.overhead_hits} max={max} />
                                <StatBar label="Lob" value={player.lob} max={max} />
                                <StatBar label="Serve" value={player.serve} max={max} />
                                <StatBar label="Grundslag" value={player.groundstrokes} max={max} />
                            </div>
                        </div>
                    );
                })}
            </div>
                </div>
            </Animation>
        </>
    );
};

export default HitsPage;