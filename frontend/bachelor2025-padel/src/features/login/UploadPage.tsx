import ParticleBackground from "./ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";
import {ArrowUpOnSquareStackIcon} from "@heroicons/react/24/outline";
import {useState} from "react";
import {useNavigate} from "react-router-dom";

const UploadPage = () => {

    const navigate = useNavigate();
    const [fileName, setFileName] = useState<string | null>(null);

    function handleAnalyse() {
        alert("Analyse started for file: " + fileName);
        navigate("/1234");
    }

    return (
        <>
            <title>Upload kamp</title>

            <div className="relative min-h-screen bg-black text-white overflow-hidden">
                <ParticleBackground />

                <Animation>
                <div className="relative z-10 h-screen flex flex-col justify-center items-center">
                    <div className="flex flex-col gap-2 justify-center items-center mb-8">
                        <h1 className="text-6xl">
                            Upload kamp
                        </h1>
                        <h2 className="text-2xl">
                            Kom i gang med at analysér din kamp
                        </h2>
                    </div>

                    <div className="flex flex-col gap-2 justify-center items-center">

                        <label
                            htmlFor="upload"
                            className="rounded border border-gray-300 shadow-sm p-6 hover:cursor-pointer hover:bg-gray-700 transition-colors"
                        >
                            <div className="flex items-center justify-center gap-4 w-64">
                                <h1 className="text-xl">Upload din kamp</h1>
                                <ArrowUpOnSquareStackIcon className="h-6 w-6 text-gray-500 dark:text-gray-400" />
                            </div>

                            <input type="file" id="upload" className="sr-only" /*accept="video/*"*/ onChange={(e) => {
                                const f = e.target.files?.[0];
                                setFileName(f ? f.name : null);
                            }}/>
                        </label>

                        <p className="text-sm text-white/70" aria-live="polite">
                            {fileName ? fileName : "Ingen fil valgt endnu"}
                        </p>

                        <button onClick={handleAnalyse} className={!fileName ? "hidden" : "text-2xl mt-6 bg-gradient-to-r from-green-300 to-blue-500 rounded-md p-4 px-8 w-64 hover:from-green-400 hover:to-blue-600 hover:cursor-pointer transition-colors"}>
                            Vamos
                        </button>
                    </div>

                </div>
                </Animation>
            </div>
        </>
    );
};

export default UploadPage;