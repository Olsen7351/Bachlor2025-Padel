import { Navigate, Outlet } from "react-router-dom";
import { onAuthStateChanged } from "firebase/auth";
import { useEffect, useState } from "react";
import {auth} from "../../firebase/firebase.ts";

export default function ProtectedRoute() {
    const [loading, setLoading] = useState(true);
    const [authorized, setAuthorized] = useState(false);

    useEffect(() => {
        const unsub = onAuthStateChanged(auth, async (user) => {
            if (!user) {
                localStorage.removeItem("idToken");
                setAuthorized(false);
                setLoading(false);
                return;
            }

            try {
                const token = await user.getIdToken(true);
                localStorage.setItem("idToken", token);
                setAuthorized(true);
            } catch (e) {
                console.error("Token refresh failed:", e);
                localStorage.removeItem("idToken");
                setAuthorized(false);
            } finally {
                setLoading(false);
            }
        });

        return () => unsub();
    }, []);

    if (loading) {
        return null;
    }

    if (!authorized) {
        return <Navigate to="/" replace />;
    }

    return <Outlet />;
}
