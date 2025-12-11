import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";

const AnalysisDashboard = () => {
    return (
        <>
        <title>Login</title>
            <div className="relative min-h-screen overflow-hidden flex flex-col gap-10">
                <ParticleBackground />

                <Animation>
                <div className="text-6xl text-center">
                    Hvad vil du udforske?
                </div>

                <div className="grid grid-cols-3 gap-6 p-4">
                    <div className="h-[calc(100vh-200px)] border w-full rounded-xl hover:bg-gray-600 transition cursor-pointer">Hej</div>
                    <div className="h-full border w-full rounded-xl hover:bg-gray-600 transition cursor-pointer">Hej</div>
                    <div className="h-full border w-full rounded-xl hover:bg-gray-600 transition cursor-pointer">Hej</div>
                </div>

                </Animation>
        </div>
        </>
    );
};

export default AnalysisDashboard;