import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";

const API_BASE = "http://localhost:8000/api";

type Video = {
    id: string;
    file_name: string;
    status: string;
    upload_timestamp: string;
    video_length: number;
};

function formatDuration(seconds: number) {
    if (!Number.isFinite(seconds) || seconds < 0) return "Ukendt";
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;

    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
    return `${m}:${String(r).padStart(2, "0")}`;
}

function formatDate(iso: string) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "Ukendt";
    return d.toLocaleString();
}

const DashboardPage = () =>{
    const navigate = useNavigate();

    const [videos, setVideos] = useState<Video[] | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const idToken = useMemo(() => localStorage.getItem("idToken"), []);

    const [page, setPage] = useState(1);
    const pageSize = 10;
    const total = videos?.length ?? 0;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));

    const pagedVideos = (videos ?? []).slice((page - 1) * pageSize, page * pageSize);

    useEffect(() => {
        if (page > totalPages) setPage(totalPages);
        if (page < 1) setPage(1);
    }, [totalPages, page]);


    useEffect(() => {
        async function fetchVideos() {
            setIsLoading(true);
            setError(null);

            try {
                const res = await fetch(`${API_BASE}/videos/analyzed`, {
                    method: "GET",
                    headers: {
                        Authorization: `Bearer ${idToken}`,
                    },
                });

                const data = await res.json().catch(() => null);

                if (!res.ok) {
                    if (res.status === 404) {
                        setVideos([]);
                        return;
                    }
                    const msg = data?.detail ?? `Kunne ikke hente kampe (${res.status})`;
                    throw new Error(msg);
                }

                const list: Video[] = Array.isArray(data) ? data : (data?.videos ?? []);
                const sortedList = list.sort((a, b) =>
                    new Date(b.upload_timestamp).getTime() - new Date(a.upload_timestamp).getTime());
                setVideos(sortedList);
                setPage(1);
            } catch (e: any) {
                if (e?.name === "AbortError") return;
                setError(e?.message ?? "Noget gik galt ved hentning af kampe.");
                setVideos([]);
            } finally {
                setIsLoading(false);
            }
        }

        fetchVideos().then();
    }, [idToken, navigate]);

    const statusBadge = (status: string) => {
        const s = (status ?? "").toLowerCase();
        const base = "text-sm border rounded-full px-3 py-1 whitespace-nowrap";

        if (["done", "completed", "ready"].includes(s)) return `${base} border-green-300`;
        if (["processing", "running", "queued", "uploading"].includes(s)) return `${base} border-blue-300`;
        if (["failed", "error"].includes(s)) return `${base} border-red-300`;
        return `${base} border-gray-300`;
    };

    return (
        <div className="relative min-h-screen overflow-hidden">
            <ParticleBackground />

            <Animation>
                <div className="relative z-10 min-h-screen flex flex-col items-center px-6 py-10">
                    <div className="w-full max-w-9xl flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-5xl">Dashboard</h1>
                        </div>

                        <div>
                            <button
                                onClick={() => navigate("/upload")}
                                className="border border-gray-300 rounded-md px-4 py-2 hover:scale-110 transition cursor-pointer"
                            >
                                Upload kamp
                            </button>
                        </div>
                    </div>

                    <div className="w-full max-w-8xl">
                        {isLoading && (
                            <div className="border border-gray-200 rounded-xl p-6 backdrop-blur-xs">
                                Henter kampe...
                            </div>
                        )}

                        {!isLoading && error && (
                            <div className="border border-red-200 rounded-xl p-6 backdrop-blur-xs text-red-700">
                                {error}
                            </div>
                        )}

                        {!isLoading && !error && videos && videos.length === 0 && (
                            <div className="border border-gray-200 rounded-xl p-10 backdrop-blur-xs text-center">
                                <h2 className="text-2xl mb-2">Ingen kampe endnu</h2>
                                <p className="text-gray-600 mb-6">
                                    Upload en kamp for at komme i gang.
                                </p>
                                <button
                                    onClick={() => navigate("/upload")}
                                    className="bg-gradient-to-r from-green-300 to-blue-500 rounded-md px-6 py-3 hover:from-green-400 hover:to-blue-600 transition-colors"
                                >
                                    Upload din første kamp
                                </button>
                            </div>
                        )}

                        {!isLoading && !error && videos && videos.length > 0 && (
                            <>
                            <div className="w-full flex items-center justify-between mb-3">
                                <div className="text-sm">
                                    Viser {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} af {total}
                                </div>

                                <div className="flex items-center gap-3">
                                    <button
                                        className="border border-gray-300 rounded-md px-3 py-1 disabled:opacity-50"
                                        disabled={page === 1}
                                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                                    >
                                        Forrige
                                    </button>

                                    <div className="text-sm">
                                        Side {page} / {totalPages}
                                    </div>

                                    <button
                                        className="border border-gray-300 rounded-md px-3 py-1 disabled:opacity-50"
                                        disabled={page === totalPages}
                                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                                    >
                                        Næste
                                    </button>
                                </div>
                            </div>


                            <div className="border border-gray-200 rounded-xl backdrop-blur-xs overflow-hidden">
                                <div className="grid grid-cols-12 gap-4 px-6 py-4 border-b border-gray-200 text-sm">
                                    <div className="col-span-4">Filnavn</div>
                                    <div className="col-span-2">Status</div>
                                    <div className="col-span-3">Uploadet</div>
                                    <div className="col-span-1 text-right">Længde</div>
                                    <div className="col-span-2 text-right">Del</div>
                                </div>

                                {pagedVideos.map((v) => (
                                    <div
                                        onClick={() => navigate(`/matches/${v.id}/overview`)}
                                        key={v.id}
                                        className="grid grid-cols-12 gap-4 px-6 py-4
                                        cursor-pointer hover:bg-gray-900 transition
                                        border-b border-gray-100 items-center"
                                    >
                                        <div className="col-span-4">
                                            <div className="font-medium">{v.file_name}</div>
                                        </div>

                                        <div className="col-span-2">
                                            <span className={statusBadge(v.status)}>{v.status}</span>
                                        </div>

                                        <div className="col-span-3">
                                            {formatDate(v.upload_timestamp)}
                                        </div>

                                        <div className="col-span-1 text-right">
                                            {formatDuration(v.video_length)}
                                        </div>

                                        <div
                                            className="col-span-2 text-right"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                navigator.clipboard.writeText(`/matches/${v.id}/overview`)
                                                    .then(() => alert("Link kopieret!"));
                                            }}
                                        >
                                            🔗
                                        </div>
                                    </div>
                                ))}
                            </div>
                            </>
                        )}
                    </div>
                </div>
            </Animation>
        </div>
    );
}

export default DashboardPage;