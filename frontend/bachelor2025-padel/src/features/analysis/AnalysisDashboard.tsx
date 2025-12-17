import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";
import {useNavigate} from "react-router-dom";
import duelsPicture from "../../assets/duels.jpg"
import hitsPicture from "../../assets/hits.png"
import heatmapPicture from "../../assets/heatmap.png"

const AnalysisDashboard = () => {
    const navigate = useNavigate();
    return (
        <>
                <ParticleBackground />
                <Animation>

                    <button className="z-30 absolute top-4 right-4 cursor-pointer border border-white rounded-xl p-3"
                            onClick={() => navigate("/upload")}>
                        Upload ny kamp
                    </button>


                    <div className="relative min-h-screen overflow-hidden flex flex-col justify-center px-10">
                        <div className="text-6xl text-center mb-10">
                            Hvad vil du udforske?
                        </div>

                <div className="grid grid-cols-3 gap-16 p-4 text-center text-4xl text-bold">
                    <div onClick={() => navigate("duels")}
                        className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
                        <img
                            src={duelsPicture}
                            alt="Duels Picture"
                            className="absolute inset-0 h-full w-full object-cover z-0"
                        />

                        <div className="absolute inset-0 bg-black/40 z-10 group-hover:bg-black/0 transition" />

                        <div className="relative z-20 flex items-center justify-center h-full">
                            <h1 className="text-4xl font-bold tracking-wide">
                                DUELLER
                            </h1>
                        </div>
                    </div>

                     <div
                            onClick={()  => navigate("hits")}
                         className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
                            <img
                                src={hitsPicture}
                                alt="Hits Picture"
                                className="absolute inset-0 h-full w-full object-cover z-0"
                            />

                            <div className="absolute inset-0 bg-black/40 z-10 group-hover:bg-black/0 transition" />

                            <div className="relative z-20 flex items-center justify-center h-full">
                                <h1 className="text-4xl font-bold tracking-wide">
                                    SLAG
                                </h1>
                            </div>
                        </div>

                        <div
                            onClick={()  => navigate("heatmap")}
                            className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
                            <img
                                src={heatmapPicture}
                                alt="Hits Picture"
                                className="absolute inset-0 h-full w-full object-cover z-0 scale-200 -translate-y-14"
                            />

                            <div className="absolute inset-0 bg-black/40 z-10 group-hover:bg-black/0 transition" />

                            <div className="relative z-20 flex items-center justify-center h-full">
                                <h1 className="text-4xl font-bold tracking-wide">
                                    HEATMAP
                                </h1>
                            </div>
                        </div>

                </div>
        </div>

                </Animation>
        </>
    );
};

export default AnalysisDashboard;