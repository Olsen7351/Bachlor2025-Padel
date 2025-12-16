import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";
import duelsPicture from "../../assets/duels.jpg";
import hitsPicture from "../../assets/hits.png";
import heatmapPicture from "../../assets/heatmap.png";

const DuelsPage = () => {
    return (
        <>
            <div className="relative min-h-screen overflow-hidden flex flex-col justify-center px-10">
                <ParticleBackground />
                <Animation>

                    <div className="text-6xl text-center mb-10">
                        Hvad vil du udforske?
                    </div>

                    <div className="grid grid-cols-3 gap-16 p-4 text-center text-4xl text-bold">
                        <div className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
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

                        <div className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
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

                        <div className="relative h-[calc(100vh-500px)] w-full rounded-xl overflow-hidden cursor-pointer group hover:scale-110 transition">
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

                </Animation>
            </div>
        </>
    );
};

export default DuelsPage;