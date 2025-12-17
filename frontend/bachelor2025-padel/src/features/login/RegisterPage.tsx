import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import {auth} from "../../../firebase/firebase.ts";
import ParticleBackground from "../../globalComponents/ParticleBackground.tsx";
import Animation from "../../globalComponents/Animation.tsx";

export default function RegisterPage() {
    const navigate = useNavigate();

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleRegister = async (event: FormEvent) => {
        event.preventDefault();
        setError(null);

        if (password !== confirmPassword) {
            setError("Adgangskoderne matcher ikke.");
            return;
        }

        setIsSubmitting(true);

        try {
            const cred = await createUserWithEmailAndPassword(
                auth,
                email.trim(),
                password
            );

            await updateProfile(cred.user, {
                displayName: email.split("@")[0],
            });
            navigate("/");
        } catch (e: any) {
            switch (e.code) {
                case "auth/email-already-in-use":
                    setError("Denne email er allerede i brug.");
                    break;
                case "auth/weak-password":
                    setError("Adgangskoden skal være mindst 6 tegn.");
                    break;
                case "auth/invalid-email":
                    setError("Ugyldig email-adresse.");
                    break;
                default:
                    setError("Noget gik galt. Prøv igen.");
            }
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="relative min-h-screen overflow-hidden">
            <ParticleBackground />

            <Animation>
                <div className="relative z-10 h-screen flex flex-col gap-4 justify-center items-center">
                    <div className="flex flex-col gap-2 justify-center items-center mb-8">
                        <h1 className="text-6xl">Opret bruger</h1>
                        <h2 className="text-2xl">Kom i gang med ViborAI</h2>
                    </div>

                    <form
                        className="flex flex-col gap-6 justify-center items-center"
                        onSubmit={handleRegister}
                    >
                        <input
                            type="email"
                            placeholder="Email"
                            className="border border-gray-300 rounded-md p-2 w-64"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />

                        <input
                            type="password"
                            placeholder="Adgangskode"
                            className="border border-gray-300 rounded-md p-2 w-64"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />

                        <input
                            type="password"
                            placeholder="Gentag adgangskode"
                            className="border border-gray-300 rounded-md p-2 w-64"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                        />

                        {error && (
                            <div className="w-64 text-red-600 text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={isSubmitting}
                            className="bg-gradient-to-r from-green-300 to-blue-500 rounded-md p-2 w-64 hover:from-green-400 hover:to-blue-600 transition-colors disabled:opacity-60"
                        >
                            {isSubmitting ? "Opretter bruger..." : "Registrer"}
                        </button>
                    </form>

                    <h1
                        onClick={() => navigate("/")}
                        className="text-gray-500 cursor-pointer"
                    >
                        Har du allerede en bruger? Log ind
                    </h1>
                </div>
            </Animation>
        </div>
    );
}
