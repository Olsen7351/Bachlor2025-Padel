import type {FormEvent} from "react";
import ParticleBackground from "./ParticleBackground.tsx";
import { useNavigate } from "react-router-dom";
import Animation from "../../globalComponents/Animation.tsx";

const LoginPage = () => {
    const navigate = useNavigate();
    let loginSuccessful = false;

    const handleLogin = (event: FormEvent) => {
        event.preventDefault();
        alert("Login form submitted");

        loginSuccessful = true;
        if (loginSuccessful) {
            navigate("/upload");
        }
    }

    return (
        <>
        <title>Login</title>

            <div className="relative min-h-screen overflow-hidden">
            <ParticleBackground />

            <Animation>
            <div className="relative z-10 h-screen flex flex-col justify-center items-center">
            <div className="flex flex-col gap-2 justify-center items-center mb-8">
                <h1 className="text-6xl">
                    Velkommen til ViborAI
                </h1>
                <h2 className="text-2xl">
                    Padelkampe på et nyt niveau
                </h2>
            </div>

                <form className="flex flex-col gap-6 justify-center items-center" onSubmit={handleLogin}>

                    <label htmlFor="brugernavn">
                        <input type="text" id="brugernavn" name="brugernavn" placeholder="Brugernavn"
                               className="border border-gray-300 rounded-md p-2 w-64"/>
                    </label>

                    <label htmlFor="adgangskode">
                        <input type="password" id="adgangskode" name="adgangskode" placeholder="Adgangskode"
                               className="border border-gray-300 rounded-md p-2 w-64"/>
                    </label>


                    <button type="submit"
                            className="bg-gradient-to-r from-green-300 to-blue-500 rounded-md p-2 w-64 hover:from-green-400 hover:to-blue-600 hover:cursor-pointer transition-colors">
                        Log ind
                    </button>


                </form>

            </div>
            </Animation>
            </div>

        </>
    );
};

export default LoginPage;