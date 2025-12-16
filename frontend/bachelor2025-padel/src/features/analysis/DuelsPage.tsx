import { useParams } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import ParticleBackground from "../../globalComponents/ParticleBackground";
import Animation from "../../globalComponents/Animation";
import StatBar from "./StatsBar";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

interface RallyItem {
    rally_id: number;
    duration: number;
}

interface RalliesAnalysis {
    match_id: number;
    total_rallies: number;
    average_duration: number;
    min_duration: number;
    max_duration: number;
    rallies: RallyItem[];
}

interface DistributionItem {
    bucket: "short" | "medium" | "long" | "very_long";
    label: string;
    count: number;
    percentage: number;
}

interface RalliesDistribution {
    match_id: number;
    total_rallies: number;
    distribution: DistributionItem[];
}

const DuelsPage = () => {
    const { videoId } = useParams();

    const [analysis, setAnalysis] = useState<RalliesAnalysis | null>(null);
    const [dist, setDist] = useState<RalliesDistribution | null>(null);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;

        const fetchAll = async () => {
            try {
                setLoading(true);
                setError(null);

                const headers = {
                    Authorization: `Bearer ${localStorage.getItem("idToken")}`,
                };

                const [aRes, dRes] = await Promise.all([
                    fetch(`${API_BASE}/rallies/${videoId}/analysis`, { headers }),
                    fetch(`${API_BASE}/rallies/${videoId}/distribution`, { headers }),
                ]);

                if (!aRes.ok) throw new Error(await aRes.text());
                if (!dRes.ok) throw new Error(await dRes.text());

                const [aJson, dJson] = await Promise.all([aRes.json(), dRes.json()]);

                if (!cancelled) {
                    setAnalysis(aJson);
                    setDist(dJson);
                }
            } catch (e: any) {
                if (!cancelled) setError(e?.message ?? "Kunne ikke hente dueller");
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        fetchAll().then();
        return () => {
            cancelled = true;
        };
    }, [videoId]);

    const maxBucketCount = useMemo(() => {
        const counts = dist?.distribution?.map((x) => x.count) ?? [];
        return Math.max(1, ...counts);
    }, [dist]);

    const formatSec = (s: number) => `${s.toFixed(2)}s`;

    return (
        <>
            <div className="relative min-h-screen bg-black text-white overflow-hidden">
                <ParticleBackground />

                <Animation>
                    <div className="relative z-10 min-h-screen flex flex-col justify-center px-10 py-16">
                        <h1 className="text-5xl font-bold text-center mb-10">DUELLER</h1>

                        {loading && (
                            <p className="text-center text-white/70">Indlæser…</p>
                        )}

                        {error && (
                            <p className="text-center text-red-300 max-w-2xl mx-auto">
                                {error}
                            </p>
                        )}

                        {!loading && !error && analysis && dist && (
                            <div className="max-w-5xl w-full mx-auto space-y-8">
                                {/* Summary cards */}
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                    <div className="rounded-xl border border-white/10 backdrop-blur-sm p-5">
                                        <div className="text-sm text-white/60">Antal dueller</div>
                                        <div className="text-3xl font-semibold mt-1">
                                            {analysis.total_rallies}
                                        </div>
                                    </div>

                                    <div className="rounded-xl border border-white/10 backdrop-blur-sm p-5">
                                        <div className="text-sm text-white/60">Gns. længde</div>
                                        <div className="text-3xl font-semibold mt-1">
                                            {formatSec(analysis.average_duration)}
                                        </div>
                                    </div>

                                    <div className="rounded-xl border border-white/10 backdrop-blur-sm p-5">
                                        <div className="text-sm text-white/60">Korteste</div>
                                        <div className="text-3xl font-semibold mt-1">
                                            {formatSec(analysis.min_duration)}
                                        </div>
                                    </div>

                                    <div className="rounded-xl border border-white/10 backdrop-blur-sm p-5">
                                        <div className="text-sm text-white/60">Længste</div>
                                        <div className="text-3xl font-semibold mt-1">
                                            {formatSec(analysis.max_duration)}
                                        </div>
                                    </div>
                                </div>

                                {/* Distribution */}
                                <div className="rounded-xl border border-white/10 backdrop-blur-sm p-6">
                                    <h2 className="text-2xl font-semibold mb-6 text-center">
                                        Fordeling
                                    </h2>

                                    <div className="space-y-4">
                                        {dist.distribution.map((b) => (
                                            <div key={b.bucket} className="space-y-2">
                                                <StatBar
                                                    label={`${b.label}  •  ${b.percentage}%`}
                                                    value={b.count}
                                                    max={maxBucketCount}
                                                />
                                                <div className="text-xs text-white/50 text-right">
                                                    {b.count} dueller
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </Animation>
            </div>
        </>
    );
};

export default DuelsPage;