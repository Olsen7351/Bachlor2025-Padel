import ParticleBackground from "../../globalComponents/ParticleBackground";
import Animation from "../../globalComponents/Animation";
import { ArrowUpOnSquareStackIcon } from "@heroicons/react/24/outline";
import {useEffect, useState} from "react";
import { useNavigate } from "react-router-dom";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

const UploadPage = () => {
    const navigate = useNavigate();

    const [file, setFile] = useState<File | null>(null);
    const [courtNumber, setCourtNumber] = useState<number>(1);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);


    useEffect(() => {
        fetch(`${API_BASE}/auth/me`, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("idToken")}`,
            }
        }).then((res) => {
            if (res.status === 401) {
                navigate("/");
            }
        })
    }, []);

    async function waitForMatchOverview(matchId: number) {
        const start = Date.now();
        const timeoutMs = 30 * 60_000;
        const intervalMs = 2 * 60_000;

        while (Date.now() - start < timeoutMs) {
            const res = await fetch(`${API_BASE}/matches/${matchId}/overview`, {
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("idToken")}`,
                },
            });

            if (res.ok) return true;

            if (res.status === 404) {
                await new Promise((r) => setTimeout(r, intervalMs));
                continue;
            }

            const text = await res.text();
            throw new Error(text || `Fejl under analyse (${res.status})`);
        }

        throw new Error("Analysen tog for lang tid. Prøv igen senere.");
    }



    async function handleAnalysis() {
        if (!file) return;

        setIsUploading(true);
        setError(null);

        try {
            const form = new FormData();
            form.append("file", file);
            form.append("court_number", String(courtNumber));

            const res = await fetch(`${API_BASE}/videos/upload`, {
                method: "POST",
                body: form,
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("idToken")}`,
                },
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Upload fejlede: ${text}`);
            }

            const upload = await res.json();
            const videoId = upload.id as number;

            await waitForMatchOverview(videoId);

            navigate(`/matches/${videoId}/overview`);
        } catch (e: any) {
            setError(e?.message ?? "Upload fejlede");
        } finally {
            setIsUploading(false);
        }
    }



    return (
        <>
            <div className="relative min-h-screen bg-black text-white overflow-hidden">
                <ParticleBackground />

                <Animation>
                    <div className="relative z-10 h-screen flex flex-col justify-center items-center">
                        <div className="flex flex-col gap-2 justify-center items-center mb-8">
                            <h1 className="text-6xl">Upload kamp</h1>
                            <h2 className="text-2xl">Kom i gang med at analysér din kamp</h2>
                        </div>

                        <div className="flex flex-col gap-3 justify-center items-center">
                            <label
                                htmlFor="upload"
                                className="rounded-md backdrop-blur-xs border border-gray-700 shadow-sm p-6 hover:cursor-pointer hover:scale-110 transition"
                            >
                                <div className="flex items-center justify-center gap-4 w-64">
                                    <h1 className="text-xl">Upload din kamp</h1>
                                    <ArrowUpOnSquareStackIcon className="h-6 w-6 text-white" />
                                </div>

                                <input
                                    type="file"
                                    id="upload"
                                    className="sr-only"
                                    disabled={isUploading}
                                    accept="video/*"
                                    onChange={(e) => {
                                        const f = e.target.files?.[0] ?? null;
                                        setFile(f);
                                    }}
                                />
                            </label>

                            <p className="text-sm text-white/70" aria-live="polite">
                                {file ? file.name : "Ingen fil valgt endnu"}
                            </p>

                            <div className="flex items-center gap-2 mt-2">
                                <span className="text-white/80">Banenummer</span>
                                <input
                                    type="number"
                                    min={1}
                                    value={courtNumber}
                                    onChange={(e) => setCourtNumber(Number(e.target.value))}
                                    className="w-20 rounded bg-black border border-white/20 px-2 py-1"
                                />
                            </div>


                            {error && (
                                <p className="text-sm text-red-300 max-w-md text-center mt-2">
                                    {error}
                                </p>
                            )}

                            <button
                                onClick={handleAnalysis}
                                disabled={!file || isUploading}
                                className={
                                    !file
                                        ? "hidden"
                                        : `hover:scale-110 text-2xl mt-6 rounded-md p-2 px-8 w-48 border border-gray-700 backdrop-blur-xs cursor-pointer transition
                                        ${isUploading ? "cursor-none animate-bounce" : ""}`
                                }
                            >
                                {isUploading ? "Analyserer..." : "Vamos"}
                            </button>
                        </div>
                    </div>
                </Animation>
            </div>
        </>
    );
}

export default UploadPage;