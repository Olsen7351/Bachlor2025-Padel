import type { FormEvent } from "react";
import { useState } from "react";
import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import { useNavigate } from "react-router-dom";
import Animation from "../../globalComponents/Animation.tsx";
import {signInWithEmailAndPassword } from "firebase/auth";
import {auth} from "../../../firebase/firebase.ts";

const API_BASE = "http://localhost:8000/api";

const LoginPage = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function firebaseLogin(email: string, password: string) {
        const cred = await signInWithEmailAndPassword(auth, email, password);
        const user = cred.user;
        const idToken = await user.getIdToken();
        return { user, idToken };
    }

    async function backendLogin(idToken: string) {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${idToken}`,
            },
        });

        const data = await res.json().catch(() => null);

        if (!res.ok) {
            const msg = data?.detail ?? `Login failed (${res.status})`;
            throw new Error(msg);
        }

        return data as { message: string; user: any };
    }

    const handleLogin = async (event: FormEvent) => {
        event.preventDefault();
        setError(null);
        setIsSubmitting(true);

        try {
            const { idToken } = await firebaseLogin(email, password);
            await backendLogin(idToken);
            navigate("/upload");
        } catch (e: any) {
            setError("Forkert email eller adgangskode.");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <>
            <title>Login</title>

            <div className="relative min-h-screen overflow-hidden">
                <ParticleBackground />

                <Animation>
                    <div className="relative z-10 h-screen flex flex-col justify-center items-center">
                        <div className="flex flex-col gap-2 justify-center items-center mb-8">
                            <h1 className="text-6xl">Velkommen til ViborAI</h1>
                            <h2 className="text-2xl">Padelkampe på et nyt niveau</h2>
                        </div>

                        <form className="flex flex-col gap-6 justify-center items-center" onSubmit={handleLogin}>
                            <label htmlFor="email" className="w-64">
                                <input
                                    type="email"
                                    id="email"
                                    name="email"
                                    placeholder="Email"
                                    className="border border-gray-300 rounded-md p-2 w-full"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                            </label>

                            <label htmlFor="password" className="w-64">
                                <input
                                    type="password"
                                    id="password"
                                    name="password"
                                    placeholder="Adgangskode"
                                    className="border border-gray-300 rounded-md p-2 w-full"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </label>

                            {error && (
                                <div className="w-64 text-red-600 text-sm">
                                    {error}
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={isSubmitting}
                                className="bg-gradient-to-r from-green-300 to-blue-500 rounded-md p-2 w-64 hover:from-green-400 hover:to-blue-600 hover:cursor-pointer transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                            >
                                {isSubmitting ? "Logger ind..." : "Log ind"}
                            </button>
                        </form>
                    </div>
                </Animation>
            </div>
        </>
    );
};

export default LoginPage;